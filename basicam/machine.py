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


@dataclass(frozen=True)
class GrblCapabilities:
    version: str
    options: str = ""
    axes: str = "XYZ"


def available_serial_ports() -> list[tuple[str, str]]:
    return [(port.device, port.description or port.device) for port in list_ports.comports()]


def parse_grbl_status(line: str) -> GrblStatus | None:
    if not (line.startswith("<") and line.endswith(">")):
        return None
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
    return GrblStatus(state, machine_position, work_position, pins, feed_rate, spindle_rpm, work_offset)


def parse_grbl_capabilities(lines: list[str]) -> GrblCapabilities | None:
    version = ""
    options = ""
    axes = "XYZ"
    for line in lines:
        if line.startswith("Grbl "):
            version = line.removeprefix("Grbl ").strip()
        version_match = re.search(r"\[VER:([^\]]+)\]", line, re.IGNORECASE)
        if version_match:
            version = version_match.group(1).strip()
        options_match = re.search(r"\[OPT:([^\]]+)\]", line, re.IGNORECASE)
        if options_match:
            options = options_match.group(1).strip()
        axes_match = re.search(r"\[AXS:\d+:([^\]]+)\]", line, re.IGNORECASE)
        if axes_match:
            axes = axes_match.group(1).strip().upper()
    return GrblCapabilities(version, options, axes) if version else None


class GrblController:
    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self.port = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0,
            write_timeout=1,
        )
        self._receive_buffer = bytearray()

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
        if command not in {b"?", b"!", b"~", b"\x18"}:
            raise ValueError("Unsupported GRBL real-time command.")
        self.port.write(command)

    def jog(self, axis: str, distance_mm: float, feed_mm_min: float) -> None:
        axis = axis.upper()
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("Only X, Y, and Z jogs are supported.")
        if not 0 < abs(distance_mm) <= 25:
            raise ValueError("Jog distance must be greater than 0 and no more than 25 mm.")
        if not 1 <= feed_mm_min <= 5000:
            raise ValueError("Jog feed must be between 1 and 5000 mm/min.")
        self.send_line(f"$J=G91 G21 {axis}{distance_mm:.4f} F{feed_mm_min:.1f}")

    def read_lines(self) -> list[str]:
        if not self.is_open:
            return []
        available = self.port.in_waiting
        if available:
            self._receive_buffer.extend(self.port.read(available))
        lines: list[str] = []
        while b"\n" in self._receive_buffer:
            line, _, remainder = self._receive_buffer.partition(b"\n")
            self._receive_buffer = bytearray(remainder)
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded:
                lines.append(decoded)
        return lines

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


def describe_grbl_message(line: str) -> str | None:
    """'ALARM:1' / 'error:22' -> readable text, else None."""
    text = line.strip()
    kind, _sep, code = text.partition(":")
    if not code.strip().isdigit():
        return None
    number = int(code.strip())
    if kind.upper() == "ALARM":
        return f"ALARM {number}: {GRBL_ALARMS.get(number, 'Unknown alarm')}"
    if kind.lower() == "error":
        return f"Error {number}: {GRBL_ERRORS.get(number, 'Unknown error')}"
    return None
