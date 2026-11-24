from __future__ import annotations

from dataclasses import dataclass
import re

import serial
from serial.tools import list_ports


@dataclass(frozen=True)
class GrblStatus:
    state: str
    machine_position: tuple[float, float, float] | None = None
    work_position: tuple[float, float, float] | None = None
    pins: str = ""
    feed_rate: float | None = None
    spindle_rpm: float | None = None
    work_offset: tuple[float, float, float] | None = None  # WCO (sent only every few reports)
    axis_values: tuple[tuple[str, float], ...] = ()  # every MPos axis by name (X Y Z A B C U V)

    def axis(self, name: str) -> float | None:
        return dict(self.axis_values).get(name)


@dataclass(frozen=True)
class GrblCapabilities:
    version: str
    options: str = ""
    axes: str = "XYZ"
    firmware: str = "grbl"  # "grbl", "grblhal" or "fluidnc" (v3 Block 7)

    @property
    def firmware_name(self) -> str:
        return {"grbl": "GRBL", "grblhal": "grblHAL", "fluidnc": "FluidNC"}.get(self.firmware, self.firmware)


def available_serial_ports() -> list[tuple[str, str]]:
    return [(port.device, port.description or port.device) for port in list_ports.comports()]


# USB-serial chips found on GRBL / grblHAL / FluidNC boards (by USB vendor id).
CONTROLLER_CHIPS = {
    0x1A86: "CH340 / CH341 (common on GRBL boards)",
    0x10C4: "Silicon Labs CP210x",
    0x0403: "FTDI",
    0x2341: "Arduino",
    0x2A03: "Arduino",
    0x303A: "Espressif ESP32 (FluidNC)",
    0x0483: "STM32 (grblHAL)",
    0x16C0: "Teensy (grblHAL)",
    0x1EAF: "STM32 / Maple (grblHAL)",
}


def serial_port_details() -> list[dict]:
    """Every serial port with its description, manufacturer, VID:PID, serial number and a hint."""
    rows = []
    for port in list_ports.comports():
        vid, pid = getattr(port, "vid", None), getattr(port, "pid", None)
        hint = CONTROLLER_CHIPS.get(vid, "") if vid is not None else ""
        text = f"{port.description or ''} {getattr(port, 'manufacturer', '') or ''}".lower()
        if not hint:
            for word, label in (("ch340", "CH340"), ("cp210", "Silicon Labs CP210x"), ("ftdi", "FTDI"),
                                ("arduino", "Arduino"), ("esp32", "Espressif ESP32 (FluidNC)")):
                if word in text:
                    hint = label
                    break
        rows.append({
            "device": port.device,
            "description": port.description or "",
            "manufacturer": getattr(port, "manufacturer", "") or "",
            "vid_pid": f"{vid:04X}:{pid:04X}" if vid is not None and pid is not None else "",
            "serial_number": getattr(port, "serial_number", "") or "",
            "hint": hint,
        })
    return rows


def parse_grbl_status(line: str, axis_names: str = "XYZABC") -> GrblStatus | None:
    """GRBL / grblHAL / FluidNC status report. MPos may carry more than three axes; they are
    named in ``axis_names`` order (from $I [AXS:...] or the machine profile)."""
    if not (line.startswith("<") and line.endswith(">")):
        return None
    axis_values: tuple[tuple[str, float], ...] = ()
    fields = line[1:-1].split("|")
    state = fields[0]
    machine_position = None
    work_position = None
    pins = ""
    feed_rate = None
    spindle_rpm = None
    work_offset = None
    for field in fields[1:]:
        key, separator, value = field.partition(":")
        if not separator:
            continue
        if key == "MPos":
            machine_position = _parse_xyz(value)
            numbers = [_parse_float(v) for v in value.split(",")]
            axis_values = tuple((n, v) for n, v in zip(axis_names, numbers) if v is not None)
        elif key == "WPos":
            work_position = _parse_xyz(value)
        elif key == "WCO":
            work_offset = _parse_xyz(value)
        elif key == "Pn":
            pins = value
        elif key == "FS":
            values = value.split(",")
            if len(values) >= 2:
                feed_rate = _parse_float(values[0])
                spindle_rpm = _parse_float(values[1])
    return GrblStatus(state, machine_position, work_position, pins, feed_rate, spindle_rpm, work_offset, axis_values)


def parse_grbl_capabilities(lines: list[str]) -> GrblCapabilities | None:
    """GRBL 1.1, grblHAL or FluidNC from the banner and ``$I`` lines; None until identified."""
    from basicam.grbl_handshake import detect_firmware

    info = detect_firmware(list(lines))
    if info is None:
        return None
    return GrblCapabilities(info.version, info.options, info.axes or "XYZ", info.kind)


class GrblController:
    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self.port = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0,
            write_timeout=1,
        )
        self._receive_buffer = bytearray()
        self._raw_pending = bytearray()  # bytes as received, for wrong-baud detection

    @property
    def is_open(self) -> bool:
        return bool(self.port.is_open)

    def send_line(self, command: str) -> None:
        if not self.is_open:
            raise RuntimeError("The serial controller is not connected.")
        encoded = command.strip().encode("ascii")
        if not encoded:
            return
        if b"\n" in encoded or b"\r" in encoded:
            raise ValueError("Send one G-code or GRBL command at a time.")
        self.port.write(encoded + b"\n")

    def send_realtime(self, command: bytes) -> None:
        if not self.is_open:
            raise RuntimeError("The serial controller is not connected.")
        if command not in {b"?", b"!", b"~", b"\x18", b"\x85"}:  # 0x85: jog cancel
            raise ValueError("Unsupported GRBL real-time command.")
        self.port.write(command)

    def jog(self, axis: str, distance_mm: float, feed_mm_min: float, axes: str = "XYZ",
            max_distance: float = 25.0) -> None:
        """One `$J=` jog. ``axes``: the axes the controller reports ([AXS:]); ``max_distance``: the
        largest single jog for this axis (the machine travel; the motion guard checks every target)."""
        axis = axis.upper()
        allowed = set(str(axes or "XYZ").upper()) & set("XYZABCUVW")
        if axis not in allowed:
            raise ValueError(f"The controller has no {axis} axis (it reports {''.join(sorted(allowed))}).")
        cap = max(float(max_distance), 0.001)
        if not 0 < abs(distance_mm) <= cap:
            raise ValueError(f"Jog distance must be greater than 0 and no more than {cap:g}.")
        if not 1 <= feed_mm_min <= 100000:
            raise ValueError("Jog feed must be between 1 and 100000 per minute.")
        self.send_line(f"$J=G91 G21 {axis}{distance_mm:.4f} F{feed_mm_min:.1f}")

    def jog_multi(self, moves: dict, feed_mm_min: float, axes: str = "XYZ", caps: dict | None = None) -> None:
        """One `$J=` line moving several axes together (pad diagonals)."""
        allowed = set(str(axes or "XYZ").upper()) & set("XYZABCUVW")
        words = []
        for axis, distance in moves.items():
            axis = axis.upper()
            if axis not in allowed:
                raise ValueError(f"The controller has no {axis} axis.")
            cap = float((caps or {}).get(axis, 25.0))
            if not 0 < abs(float(distance)) <= cap:
                raise ValueError(f"{axis} jog must be greater than 0 and no more than {cap:g}.")
            words.append(f"{axis}{float(distance):.4f}")
        if not words:
            raise ValueError("Nothing to jog.")
        if not 1 <= feed_mm_min <= 100000:
            raise ValueError("Jog feed must be between 1 and 100000 per minute.")
        self.send_line(f"$J=G91 G21 {' '.join(words)} F{feed_mm_min:.1f}")

    def read_lines(self) -> list[str]:
        if not self.is_open:
            return []
        available = self.port.in_waiting
        if available:
            chunk = self.port.read(available)
            self._receive_buffer.extend(chunk)
            self._raw_pending.extend(chunk)
            del self._raw_pending[:-4096]
        lines: list[str] = []
        while b"\n" in self._receive_buffer:
            line, _, remainder = self._receive_buffer.partition(b"\n")
            self._receive_buffer = bytearray(remainder)
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded:
                lines.append(decoded)
        return lines

    def read_raw(self) -> bytes:
        """Bytes received since the last call (decoded lines still come from read_lines)."""
        data = bytes(self._raw_pending)
        self._raw_pending.clear()
        return data

    def send_blank(self) -> None:
        """A bare newline: wakes a controller that did not reset when the port opened."""
        if not self.is_open:
            raise RuntimeError("The serial controller is not connected.")
        self.port.write(b"\n")

    @property
    def port_name(self) -> str:
        return str(getattr(self.port, "port", "") or "")

    def close(self) -> None:
        if self.is_open:
            self.port.close()


def _parse_xyz(value: str) -> tuple[float, float, float] | None:
    values = value.split(",")
    if len(values) < 3:
        return None
    try:
        return float(values[0]), float(values[1]), float(values[2])
    except ValueError:
        return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None

# GRBL 1.1 alarm and error codes (from the GRBL 1.1 interface documentation).
GRBL_ALARMS = {
    1: "Hard limit triggered; position is likely lost, re-home.",
    2: "Soft limit: the motion target exceeds machine travel.",
    3: "Reset while in motion; position may be lost, re-home.",
    4: "Probe fail: probe not in the expected initial state.",
    5: "Probe fail: probe did not contact the work.",
    6: "Homing fail: reset during homing.",
    7: "Homing fail: safety door opened during homing.",
    8: "Homing fail: pull-off did not clear the limit switch.",
    9: "Homing fail: could not find the limit switch.",
    10: "Homing fail: second switch of a dual axis not found.",
}
# grblHAL and FluidNC number alarms 10 and above differently (from the grblHAL alarm list and
# FluidNC's ExecAlarm enum); a code is looked up for the identified firmware first.
GRBLHAL_ALARMS = {
    10: "Emergency stop is active: release it, then reset.",
    11: "Homing required: home the machine ($H) before moving.",
    12: "A limit switch is engaged: move off it or check the switch wiring and inversion ($5).",
    13: "Probe protection triggered.",
    14: "Spindle did not reach speed in time.",
    15: "Homing fail: auto-squaring switch not found.",
    16: "Power-on self test failed.",
    17: "Motor fault.",
    18: "Homing fail: bad homing configuration.",
}
FLUIDNC_ALARMS = {
    10: "Spindle control fault.",
    11: "A control pin is active at start-up: check the reset, hold, door and start inputs.",
    12: "Homing fail: a switch is ambiguous (both ends active?).",
    13: "Hard stop.",
    14: "Not homed: home the machine ($H) before moving.",
    15: "Controller initialisation alarm: check the configuration.",
}
ALARM_ACTIONS = {
    1: "Check the limit switches, then Unlock ($X) and home again.",
    2: "Unlock ($X); the target was outside the travel.",
    3: "Unlock ($X) and home again.",
    9: "Check the homing switches and direction, then home again.",
    11: "Home the machine.",
    12: "Run the limit-switch test (Machine panel).",
    14: "Home the machine.",
}
GRBL_ERRORS = {
    1: "Letter not found in a G-code word.", 2: "Numeric value missing or invalid.",
    3: "'$' command not recognised.", 4: "Negative value for a positive-only setting.",
    5: "Homing is not enabled in settings.", 6: "Step pulse time must be over 3 µs.",
    7: "EEPROM read failed.", 8: "'$' command only allowed when Idle.",
    9: "G-code locked out during alarm or jog.", 10: "Soft limits need homing enabled.",
    11: "Line too long.", 12: "Setting exceeds the maximum step rate.",
    13: "Safety door opened.", 14: "Startup/build line too long.",
    15: "Jog target exceeds machine travel.", 16: "Invalid jog command.",
    17: "Laser mode requires PWM output.", 20: "Unsupported or invalid G-code command.",
    21: "Two commands from the same modal group.", 22: "Feed rate not set.",
    23: "Command needs an integer value.", 24: "Two commands need axis words.",
    25: "Repeated G-code word.", 26: "No axis words in the command.",
    27: "Invalid line number.", 28: "Command is missing a required value.",
    29: "G59.x work systems not supported.", 30: "G53 only allowed with G0 and G1.",
    31: "Axis words with no command using them.", 32: "Arc needs an in-plane axis word.",
    33: "Invalid motion target.", 34: "Invalid arc radius.",
    35: "Arc needs an in-plane offset word.", 36: "Unused value words in the block.",
    37: "Tool length offset not on the configured axis.", 38: "Tool number too high.",
}


def alarm_text(number: int, firmware: str = "grbl") -> str:
    table = {"grblhal": GRBLHAL_ALARMS, "fluidnc": FLUIDNC_ALARMS}.get(firmware, {})
    return table.get(number) or GRBL_ALARMS.get(number, "Unknown alarm")


def describe_grbl_message(line: str, firmware: str = "grbl") -> str | None:
    """'ALARM:1' / 'error:22' -> readable text, else None."""
    text = line.strip()
    kind, _sep, code = text.partition(":")
    if not code.strip().isdigit():
        return None
    number = int(code.strip())
    if kind.upper() == "ALARM":
        action = ALARM_ACTIONS.get(number, "")
        return f"ALARM {number}: {alarm_text(number, firmware)}" + (f" {action}" if action else "")
    if kind.lower() == "error":
        return f"Error {number}: {GRBL_ERRORS.get(number, 'Unknown error')}"
    return None
