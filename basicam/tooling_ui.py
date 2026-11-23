from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from basicam.tooling import (
    TOOL_TYPES,
    ToolDefinition,
    create_tool_mesh,
    default_profile,
)


class ToolLibraryDialog(QDialog):
    def __init__(self, tools: list[ToolDefinition], parent=None, machine_store=None, material_store=None, picker: bool = False) -> None:
        super().__init__(parent)
        self.picker = picker  # Assign tool: pick a tool, OK / Cancel
        self.setWindowTitle("Pick a tool" if picker else "Tool inventory")
        self.resize(1120, 720)
        self.tools = deepcopy(tools)
        self.selected_index = -1
        self.machine_store = machine_store
        self.material_store = material_store
        self.machines_tab = None
        self.materials_tab = None
        self._build_ui()
        self._refresh_list()
        if self.tools and not picker:
            self.inventory.setCurrentRow(0)
        if picker:  # nothing chosen yet: OK stays disabled until a tool is clicked
            self.inventory.setCurrentRow(-1)
            self.selected_index = -1
            self.ok_button.setEnabled(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        if self.machine_store is not None:
            from basicam.machine_profiles import MachinesTab

            self.tabs = QTabWidget()
            self.tabs.addTab(splitter, "Tools")
            self.machines_tab = MachinesTab(self.machine_store)
            self.tabs.addTab(self.machines_tab, "Machines")
            if self.material_store is not None:
                from basicam.part_materials_dialog import MaterialsTab

                self.materials_tab = MaterialsTab(self.material_store)
                self.tabs.addTab(self.materials_tab, "Materials")
            root.addWidget(self.tabs, 1)
        else:
            root.addWidget(splitter, 1)

        left = QVBoxLayout()
        left_label = QLabel("INVENTORY")
        left_label.setObjectName("sectionLabel")
        left_label.setToolTip("Saved cutters available in the tool library")
        left.addWidget(left_label)
        self.inventory = QListWidget()
        self.inventory.setToolTip("Select a tool to inspect or edit its dimensions and revolved profile")
        self.inventory.currentRowChanged.connect(self._select_tool)
        left.addWidget(self.inventory, 1)
        new_tool = QPushButton("New tool")
        new_tool.setToolTip("Start a new tool definition from the currently selected cutter type")
        new_tool.clicked.connect(self._new_tool)
        left.addWidget(new_tool)
        left_panel = QWidget()
        left_panel.setLayout(left)
        splitter.addWidget(left_panel)

        editor = QVBoxLayout()
        identity = QFormLayout()
        self.name = QLineEdit()
        self.name.setToolTip("Display name for this tool in the library and operation dropdown")
        identity.addRow("Tool name", self.name)
        self.tool_type = QComboBox()
        self.tool_type.addItems(TOOL_TYPES)
        self.tool_type.setToolTip("Cutter geometry category used to generate default profile shapes")
        identity.addRow("Cutter type", self.tool_type)
        editor.addLayout(identity)

        field_tips = {
            "diameter_mm": "Nominal cutting diameter at the widest cutting edge (mm)",
            "tip_diameter_mm": "Diameter at the very tip of the tool, such as a countersink or tap pilot (mm)",
            "included_angle_deg": "Total included angle between cutting edges for drills, chamfers, and V-bits (degrees)",
            "flute_count": "Number of cutting flutes or teeth on the cutter",
            "overall_length_mm": "Total axial length from tip (Z = 0) to the top of the shank (mm)",
            "flute_length_mm": "Axial height of the fluted/relief region along Z (mm)",
            "cutting_length_mm": "Axial engagement length of the cutting teeth/head (mm)",
            "shank_diameter_mm": "Diameter of the tool shank held in the collet or chuck (mm)",
        }

        dimensions = QFormLayout()
        self.fields: dict[str, QDoubleSpinBox | QSpinBox] = {}
        for key, title, default, upper, decimals in (
            ("diameter_mm", "Diameter · mm", 6.0, 250.0, 3),
            ("tip_diameter_mm", "Tip diameter · mm", 0.0, 250.0, 3),
            ("included_angle_deg", "Included angle · °", 118.0, 170.0, 2),
            ("flute_count", "Cutting edges / flutes", 2, 64, 0),
            ("overall_length_mm", "Overall length · mm", 60.0, 1000.0, 2),
            ("flute_length_mm", "Flute length · mm", 20.0, 500.0, 2),
            ("cutting_length_mm", "Cutting length · mm", 18.0, 500.0, 2),
            ("shank_diameter_mm", "Shank diameter · mm", 6.0, 250.0, 3),
        ):
            if key == "flute_count":
                field = QSpinBox()
                field.setRange(0, int(upper))
                field.setValue(int(default))
            else:
                field = QDoubleSpinBox()
                field.setRange(0.0, float(upper))
                field.setDecimals(int(decimals))
                field.setValue(float(default))
            field.setToolTip(field_tips[key])
            self.fields[key] = field
            dimensions.addRow(title, field)
        editor.addLayout(dimensions)

        profile_header = QHBoxLayout()
        profile_label = QLabel("REVOLVED HALF-PROFILE · RADIUS / AXIAL Z")
        profile_label.setToolTip(
            "2D cross-section points (Radius from Z axis, Axial height Z) revolved 360° around Z to build the preview mesh"
        )
        profile_header.addWidget(profile_label)
        profile_header.addStretch(1)
        defaults_button = QPushButton("Reset profile")
        defaults_button.setToolTip(
            "Rebuild the Radius / Axial Z profile table from the current cutter type and dimensions"
        )
        defaults_button.clicked.connect(self._reset_profile)
        profile_header.addWidget(defaults_button)
        add_point = QPushButton("Add point")
        add_point.setToolTip("Append a new (Radius, Axial Z) row to the revolved profile table")
        add_point.clicked.connect(self._add_profile_point)
        profile_header.addWidget(add_point)
        remove_point = QPushButton("Remove point")
        remove_point.setToolTip("Delete the currently selected (Radius, Axial Z) row from the profile table")
        remove_point.clicked.connect(self._remove_profile_point)
        profile_header.addWidget(remove_point)
        editor.addLayout(profile_header)

        self.profile_table = QTableWidget(0, 2)
        self.profile_table.setHorizontalHeaderLabels(["Radius · mm", "Axial Z · mm"])
        self.profile_table.horizontalHeader().setStretchLastSection(True)
        self.profile_table.setMaximumHeight(155)
        self.profile_table.setToolTip(
            "Each row defines a point (Radius, Axial Z) revolved around the spindle Z axis"
        )
        editor.addWidget(self.profile_table)

        actions = QHBoxLayout()
        preview_button = QPushButton("Update preview")
        preview_button.setToolTip("Revolve the current profile points and refresh the 3D cutter preview below")
        preview_button.clicked.connect(self._update_preview)
        actions.addWidget(preview_button)
        add_button = QPushButton("Add / update inventory")
        add_button.setObjectName("primaryButton")
        add_button.setToolTip("Save changes to the selected tool or add a new tool to the inventory")
        add_button.clicked.connect(self._save_tool)
        actions.addWidget(add_button)
        remove_button = QPushButton("Remove")
        remove_button.setToolTip("Remove the selected tool from the inventory list")
        remove_button.clicked.connect(self._remove_tool)
        actions.addWidget(remove_button)
        editor.addLayout(actions)

        self.preview = QtInteractor(self)
        self.preview.set_background("#161f21")
        self.preview.add_axes(line_width=2)
        self.preview.interactor.setToolTip(
            "3D viewport showing the revolved cutter envelope from the inventory profile"
        )
        editor.addWidget(self.preview.interactor, 1)
        editor_widget = QWidget()
        editor_widget.setLayout(editor)
        splitter.addWidget(editor_widget)
        splitter.setSizes([220, 850])

        footer = QHBoxLayout()
        note = QLabel("Preview shows the cutter envelope; flute helix and cutting edges are not simulated.")
        note.setWordWrap(True)
        note.setToolTip("The 3D view renders the revolved outer envelope, not helical flutes or chip load")
        footer.addWidget(note, 1)
        if self.picker:
            self.ok_button = QPushButton("OK")
            self.ok_button.setEnabled(False)  # until a tool is selected
            self.ok_button.clicked.connect(self.accept)
            cancel = QPushButton("Cancel")
            cancel.clicked.connect(self.reject)
            footer.addWidget(self.ok_button)
            footer.addWidget(cancel)
        else:
            close_button = QPushButton("Done")
            close_button.setToolTip("Close the tool inventory dialog")
            close_button.clicked.connect(self.accept)
            footer.addWidget(close_button)
        root.addLayout(footer)

        # Live preview: any edit refreshes the 3D cutter after a short pause.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(lambda: self._update_preview(silent=True))
        for field in self.fields.values():
            field.valueChanged.connect(self._preview_timer.start)
        self.tool_type.currentTextChanged.connect(self._preview_timer.start)
        self.profile_table.itemChanged.connect(self._preview_timer.start)

    def _refresh_list(self) -> None:
        current = self.selected_index
        self.inventory.blockSignals(True)
        self.inventory.clear()
        for tool in self.tools:
            self.inventory.addItem(QListWidgetItem(f"{tool.name} · {tool.diameter_mm:g} mm"))
        self.inventory.blockSignals(False)
        if self.tools:
            self.selected_index = min(max(current, 0), len(self.tools) - 1)
            self.inventory.setCurrentRow(self.selected_index)

    def selected_tool(self) -> ToolDefinition | None:
        return self.tools[self.selected_index] if 0 <= self.selected_index < len(self.tools) else None

    def _select_tool(self, index: int) -> None:
        if getattr(self, "ok_button", None) is not None:
            self.ok_button.setEnabled(0 <= index < len(self.tools))
        if not 0 <= index < len(self.tools):
            return
        self.selected_index = index
        tool = self.tools[index]
        self.name.setText(tool.name)
        self.tool_type.setCurrentText(tool.tool_type)
        for key, field in self.fields.items():
            field.setValue(getattr(tool, key))
        self._set_profile_table(tool.profile_points)
        self._update_preview()

    def _new_tool(self) -> None:
        tool_type = self.tool_type.currentText()
        tool = ToolDefinition(f"New {tool_type}", tool_type)
        self.name.setText(tool.name)
        for key, field in self.fields.items():
            field.setValue(getattr(tool, key))
        self.selected_index = -1
        self._set_profile_table(tool.profile_points)
        self._update_preview()

    def _make_tool(self) -> ToolDefinition:
        values = {key: field.value() for key, field in self.fields.items()}
        points = []
        for row in range(self.profile_table.rowCount()):
            radius_item = self.profile_table.item(row, 0)
            height_item = self.profile_table.item(row, 1)
            if not radius_item or not height_item:
                continue
            if not radius_item.text().strip() or not height_item.text().strip():
                continue
            try:
                points.append((float(radius_item.text()), float(height_item.text())))
            except ValueError as error:
                raise ValueError(f"Profile row {row + 1} must contain numeric dimensions.") from error
        return ToolDefinition(
            name=self.name.text().strip() or f"Custom {self.tool_type.currentText()}",
            tool_type=self.tool_type.currentText(),
            profile_points=points,
            **values,
        )

    def _set_profile_table(self, points: list[tuple[float, float]]) -> None:
        self.profile_table.setRowCount(len(points))
        for row, (radius, height) in enumerate(points):
            self.profile_table.setItem(row, 0, QTableWidgetItem(f"{radius:g}"))
            self.profile_table.setItem(row, 1, QTableWidgetItem(f"{height:g}"))

    def _reset_profile(self) -> None:
        try:
            tool = self._make_tool()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid profile", str(error))
            return
        self._set_profile_table(default_profile(tool))
        self._update_preview()

    def _add_profile_point(self) -> None:
        row = self.profile_table.rowCount()
        self.profile_table.insertRow(row)
        self.profile_table.setItem(row, 0, QTableWidgetItem("0"))
        self.profile_table.setItem(row, 1, QTableWidgetItem("0"))
        self.profile_table.setCurrentCell(row, 0)

    def _remove_profile_point(self) -> None:
        row = self.profile_table.currentRow()
        if row >= 0:
            self.profile_table.removeRow(row)

    def _update_preview(self, silent: bool = False) -> None:
        try:
            mesh = create_tool_mesh(self._make_tool())
        except (ValueError, TypeError, AttributeError) as error:
            if not silent:
                QMessageBox.warning(self, "Invalid tool profile", str(error))
            return
        self.preview.clear_actors()
        self.preview.add_axes(line_width=2)
        self.preview.add_mesh(mesh, color="#c2df54", show_edges=True, edge_color="#536348")
        self.preview.view_isometric()
        self.preview.reset_camera()
        self.preview.render()

    def _save_tool(self) -> None:
        try:
            tool = self._make_tool()
            create_tool_mesh(tool)
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "Invalid tool", str(error))
            return
        if self.selected_index < 0:
            self.tools.append(tool)
            self.selected_index = len(self.tools) - 1
        else:
            self.tools[self.selected_index] = tool
        self._refresh_list()
        self._update_preview()

    def _remove_tool(self) -> None:
        if not 0 <= self.selected_index < len(self.tools):
            return
        self.tools.pop(self.selected_index)
        self.selected_index = min(self.selected_index, len(self.tools) - 1)
        self._refresh_list()
        if self.tools:
            self._select_tool(self.selected_index)
        else:
            self.preview.clear_actors()
            self.preview.add_axes(line_width=2)
            self.preview.render()
