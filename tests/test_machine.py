import unittest
from unittest.mock import patch

from basicam.machine import GrblController, parse_grbl_capabilities, parse_grbl_status


class FakeSerial:
    def __init__(self) -> None:
        self.is_open = True
        self.writes: list[bytes] = []
        self.buffer = bytearray()

    @property
    def in_waiting(self) -> int:
        return len(self.buffer)

    def read(self, count: int) -> bytes:
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.is_open = False


class GrblProtocolTests(unittest.TestCase):
    def test_status_parses_machine_position_and_pins(self) -> None:
        status = parse_grbl_status("<Idle|MPos:1.000,2.500,-3.000|FS:500,0|Pn:XYZ>")
        self.assertIsNotNone(status)
        self.assertEqual(status.state, "Idle")
        self.assertEqual(status.machine_position, (1.0, 2.5, -3.0))
        self.assertEqual(status.pins, "XYZ")

    def test_capabilities_parse_version_options_and_axes(self) -> None:
        capabilities = parse_grbl_capabilities(
            ["[VER:1.1h.2024:]", "[OPT:V,15,128]", "[AXS:3:XYZ]"]
        )
        self.assertIsNotNone(capabilities)
        self.assertEqual(capabilities.version, "1.1h.2024:")
        self.assertEqual(capabilities.axes, "XYZ")

    def test_jog_command_is_limited_and_serial_lines_are_buffered(self) -> None:
        fake = FakeSerial()
        with patch("basicam.machine.serial.Serial", return_value=fake):
            controller = GrblController("COM7")
            controller.jog("X", 2.5, 300)
            self.assertEqual(fake.writes[-1], b"$J=G91 G21 X2.5000 F300.0\n")
            with self.assertRaises(ValueError):
                controller.jog("X", 26, 300)
            fake.buffer.extend(b"Grbl 1.1h\r\n<Idle|MPos:0,0,0>\r\n")
            self.assertEqual(controller.read_lines(), ["Grbl 1.1h", "<Idle|MPos:0,0,0>"])
            controller.close()


if __name__ == "__main__":
    unittest.main()