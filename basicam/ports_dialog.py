"""Machine > Ports...: every serial port with its details, refreshed while open (v3 Block 6)."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from basicam.machine_profiles import BAUD_RATES
from basicam.window_fit import make_working_window

COLUMNS = ("Port", "Description", "Manufacturer", "VID:PID", "Serial number", "Likely", "In use")


class PortsDialog(QDialog):
    """Lists ports through ``details()`` (``machine.serial_port_details`` by default). The host
    supplies ``in_use(device)``, ``test(device, baud)``, ``connect(device, baud)`` and
    ``use_for_machine(device, baud)``; the dialog itself never opens a port."""

    REFRESH_MS = 2000

    def __init__(self, parent=None, details=None, in_use=None, test=None, connect=None, use_for_machine=None,
                 baud: str = "115200") -> None:
        super().__init__(parent)
        from basicam.machine import serial_port_details

        self.setWindowTitle("Ports")
        self.setObjectName("portsWindow")
        make_working_window(self)
        self.resize(820, 360)
        self.details = details or serial_port_details
        self.in_use = in_use or (lambda _device: False)
        self.test_callback = test
        self.connect_callback = connect
        self.use_callback = use_for_machine
        self._signature: tuple = ()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Serial ports on this computer. Plug the controller in and it appears here."))
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table, 1)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        row = QHBoxLayout()
        row.addWidget(QLabel("Baud rate"))
        self.baud = QComboBox()
        self.baud.setEditable(True)
        self.baud.addItems(["Auto", *BAUD_RATES])
        self.baud.setCurrentText(str(baud))
        self.baud.setToolTip("Serial speed set in the controller firmware; Auto tries the common rates.")
        row.addWidget(self.baud)
        row.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        self.test_button = QPushButton("Test")
        self.test_button.setToolTip("Identify the firmware on this port without moving anything.")
        self.test_button.clicked.connect(self._test)
        row.addWidget(self.test_button)
        self.use_button = QPushButton("Use for this machine")
        self.use_button.setToolTip("Store this port and baud rate on the loaded machine.")
        self.use_button.clicked.connect(self._use)
        row.addWidget(self.use_button)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._connect)
        row.addWidget(self.connect_button)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        layout.addLayout(row)
        for button in (refresh, self.test_button, self.use_button, self.connect_button, close):
            button.setAutoDefault(False)
        self.timer = QTimer(self)
        self.timer.setInterval(self.REFRESH_MS)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def ports(self) -> list[dict]:
        try:
            return list(self.details())
        except Exception as error:  # noqa: BLE001
            self.result_label.setText(f"Could not list ports: {error}")
            return []

    def refresh(self) -> None:
        rows = self.ports()
        signature = tuple((r.get("device"), r.get("description"), bool(self.in_use(r.get("device")))) for r in rows)
        if signature == self._signature:
            return  # nothing changed: keep the selection
        self._signature = signature
        selected = self.selected_device()
        self.table.setRowCount(len(rows))
        for index, info in enumerate(rows):
            values = (info.get("device", ""), info.get("description", ""), info.get("manufacturer", ""),
                      info.get("vid_pid", ""), info.get("serial_number", ""), info.get("hint", ""),
                      "BasiCAM" if self.in_use(info.get("device")) else "")
            for col, value in enumerate(values):
                self.table.setItem(index, col, QTableWidgetItem(str(value)))
        if not rows:
            self.result_label.setText("No serial ports found. Check the USB cable (charge-only cables carry no data).")
        match = next((i for i, r in enumerate(rows) if r.get("device") == selected), 0 if rows else -1)
        if match >= 0:
            self.table.selectRow(match)
        self._update_buttons()

    def selected_device(self) -> str:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text() if item is not None else ""

    def baud_value(self) -> str:
        return self.baud.currentText().strip() or "115200"

    def _update_buttons(self) -> None:
        has = bool(self.selected_device())
        for button, callback in ((self.test_button, self.test_callback), (self.connect_button, self.connect_callback),
                                 (self.use_button, self.use_callback)):
            button.setEnabled(has and callback is not None)

    def _test(self) -> None:
        if self.test_callback is not None and self.selected_device():
            self.result_label.setText(f"Testing {self.selected_device()}...")
            text = self.test_callback(self.selected_device(), self.baud_value())
            if text:
                self.result_label.setText(str(text))

    def _connect(self) -> None:
        if self.connect_callback is not None and self.selected_device():
            self.connect_callback(self.selected_device(), self.baud_value())
            self._signature = ()
            self.refresh()

    def _use(self) -> None:
        if self.use_callback is not None and self.selected_device():
            text = self.use_callback(self.selected_device(), self.baud_value())
            if text:
                self.result_label.setText(str(text))
