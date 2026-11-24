from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
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
    tool_preview_mesh,
    validate_tool,
)
from basicam.theme import color as theme_color
from basicam.window_fit import make_working_window, scrollable


class ToolLibraryDialog(QDialog):
    def __init__(self, tools: list[ToolDefinition], parent=None, machine_store=None, material_store=None,
                 picker: bool = False, connect_callback=None, is_connected=None, on_change=None,
                 inches: bool = False) -> None:
        super().__init__(parent)
        self.inches = bool(inches)  # lengths shown in the display unit; stored in mm
        self._flute_linked = True  # flute length follows the cutting length until edited
        self._syncing = False
        # Called with the whole inventory after every Add / update and Remove, so tools are saved
        # at once instead of only when the window is accepted (v3 Block 3).
        self.on_change = on_change
        self._baseline = None
        self.connect_callback = connect_callback  # Machines tab: connect to a saved machine
        self.is_connected = is_connected
        self.picker = picker  # Assign tool: pick a tool, OK / Cancel
        self.setWindowTitle("Pick a tool" if picker else "Tool inventory")
        self.setObjectName("toolInventoryWindow")
        make_working_window(self)
        self.resize(1120, 720)  # preferred size; fit_to_screen clamps it to the screen on first show
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
            self.machines_tab = MachinesTab(
                self.machine_store, connect_callback=self.connect_callback, is_connected=self.is_connected
            )
            self.machines_tab.dirtyChanged.connect(self._machines_dirty_changed)
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
        self.inventory.currentRowChanged.connect(self._tool_row_changed)
        left.addWidget(self.inventory, 1)
        new_tool = QPushButton("New tool")
        new_tool.setToolTip("Start a new tool definition from the currently selected cutter type")
        new_tool.clicked.connect(self._new_tool)
        left.addWidget(new_tool)
        left_panel = QWidget()
        left_panel.setLayout(left)
        splitter.addWidget(left_panel)

        editor = QVBoxLayout()
        fields_layout = QVBoxLayout()
        identity = QFormLayout()
        self.name = QLineEdit()
        self.name.setToolTip("Display name for this tool in the library and operation dropdown")
        identity.addRow("Tool name", self.name)
        self.tool_type = QComboBox()
        self.tool_type.addItems(TOOL_TYPES)
        self.tool_type.setToolTip("Cutter geometry category used to generate default profile shapes")
        identity.addRow("Cutter type", self.tool_type)
        fields_layout.addLayout(identity)

        unit = "in" if self.inches else "mm"
        self._length_keys = {
            "diameter_mm", "shank_diameter_mm", "cutting_length_mm", "flute_length_mm", "neck_diameter_mm",
            "reach_mm", "overall_length_mm", "corner_radius_mm", "tip_diameter_mm",
        }
        # (key, label, default mm / value, upper mm / value, decimals) - lengths from the tip (Z = 0)
        specs = (
            ("diameter_mm", f"Cutting diameter ({unit})", 6.0, 250.0, 3),
            ("shank_diameter_mm", f"Shank diameter ({unit})", 6.0, 250.0, 3),
            ("cutting_length_mm", f"Cutting length ({unit})", 18.0, 500.0, 3),
            ("flute_length_mm", f"Flute length ({unit})", 18.0, 500.0, 3),
            ("neck_diameter_mm", f"Neck diameter ({unit})", 0.0, 250.0, 3),
            ("reach_mm", f"Reach ({unit})", 0.0, 1000.0, 3),
            ("overall_length_mm", f"Overall length ({unit})", 60.0, 1000.0, 3),
            ("corner_radius_mm", f"Corner radius ({unit})", 0.0, 125.0, 3),
            ("taper_angle_deg", "Taper angle per side (°)", 0.0, 30.0, 2),
            ("included_angle_deg", "Included angle (°)", 118.0, 170.0, 2),
            ("tip_diameter_mm", f"Tip diameter ({unit})", 0.0, 250.0, 3),
            ("flute_count", "Flutes", 2, 64, 0),
            ("helix_angle_deg", "Helix angle (°)", 30.0, 60.0, 1),
        )
        field_tips = {
            "cutting_length_mm": "From the tip to the end of the cutting edge.",
            "flute_length_mm": "From the tip to the end of the flute; follows the cutting length until you change it.",
            "neck_diameter_mm": "Reduced neck above the flutes (0 = no neck).",
            "reach_mm": "From the tip to the end of the neck.",
            "overall_length_mm": "From the tip to the top of the shank.",
            "taper_angle_deg": "Tapered end mills: angle of each side from the tool axis.",
            "helix_angle_deg": "Angle of the flutes from the tool axis (0 = straight flutes).",
        }
        dimensions = QFormLayout()
        self.dimensions_form = dimensions
        self.fields: dict[str, QDoubleSpinBox | QSpinBox] = {}
        self._field_labels: dict[str, str] = {}
        for key, title, default, upper, decimals in specs:
            if key == "flute_count":
                field = QSpinBox()
                field.setRange(0, int(upper))
                field.setValue(int(default))
            else:
                field = QDoubleSpinBox()
                scale = (1 / 25.4) if (self.inches and key in self._length_keys) else 1.0
                field.setDecimals(4 if scale != 1.0 else int(decimals))
                field.setRange(0.0, float(upper) * scale)
                field.setValue(float(default) * scale)
            if key in field_tips:
                field.setToolTip(field_tips[key])
            self.fields[key] = field
            self._field_labels[key] = title
            dimensions.addRow(title, field)
        self.fields["cutting_length_mm"].valueChanged.connect(self._cutting_length_changed)
        self.fields["flute_length_mm"].valueChanged.connect(self._flute_length_changed)
        self.tool_type.currentTextChanged.connect(self._apply_type_visibility)
        fields_layout.addLayout(dimensions)

        profile_header = QHBoxLayout()
        profile_label = QLabel("Custom profile (radius, height from the tip)")
        profile_label.setWordWrap(True)
        profile_header.addWidget(profile_label, 1)
        defaults_button = QPushButton("Reset profile")
        defaults_button.setToolTip("Go back to the profile generated from the dimensions above.")
        defaults_button.clicked.connect(self._reset_profile)
        profile_header.addWidget(defaults_button)
        add_point = QPushButton("Add point")
        add_point.clicked.connect(self._add_profile_point)
        profile_header.addWidget(add_point)
        remove_point = QPushButton("Remove point")
        remove_point.clicked.connect(self._remove_profile_point)
        profile_header.addWidget(remove_point)

        self.profile_table = QTableWidget(0, 2)
        self.profile_table.setHorizontalHeaderLabels(["Radius (mm)", "Height from tip (mm)"])
        self.profile_table.horizontalHeader().setStretchLastSection(True)
        self.profile_table.setMinimumHeight(60)
        self.profile_table.setToolTip("Each row is a point of the outline, turned around the tool axis.")
        # Profile table and preview share a draggable splitter instead of a fixed table height.
        # Fields (scrolling), profile table and preview share a draggable splitter, so the
        # window can shrink to any screen.
        self.editor_splitter = QSplitter(Qt.Orientation.Vertical)
        self.editor_splitter.setChildrenCollapsible(False)
        fields_widget = QWidget()
        fields_widget.setLayout(fields_layout)
        self.editor_splitter.addWidget(scrollable(fields_widget))
        profile_widget = QWidget()
        profile_box = QVBoxLayout(profile_widget)
        profile_box.setContentsMargins(0, 0, 0, 0)
        profile_box.addLayout(profile_header)
        profile_box.addWidget(self.profile_table, 1)
        self.profile_widget = profile_widget
        self.editor_splitter.addWidget(profile_widget)
        editor.addWidget(self.editor_splitter, 1)

        actions = QHBoxLayout()
        self.custom_button = QPushButton("Edit as custom")
        self.custom_button.setToolTip("Edit the outline point by point instead of generating it from the dimensions.")
        self.custom_button.clicked.connect(self._edit_as_custom)
        actions.addWidget(self.custom_button)
        self.show_dimensions = QCheckBox("Show dimensions")
        self.show_dimensions.setChecked(True)
        self.show_dimensions.setToolTip("Draw where each length and diameter is measured from in the preview.")
        self.show_dimensions.toggled.connect(lambda *_: self._update_preview(silent=True))
        actions.addWidget(self.show_dimensions)
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
        self.preview.interactor.setToolTip("Cutting edges in orange, flutes in green, neck and shank in grey.")
        self.preview.interactor.setMinimumHeight(120)
        self.editor_splitter.addWidget(self.preview.interactor)
        self.editor_splitter.setStretchFactor(0, 2)
        self.editor_splitter.setStretchFactor(1, 1)
        self.editor_splitter.setStretchFactor(2, 3)
        editor_widget = QWidget()
        editor_widget.setLayout(editor)
        splitter.addWidget(editor_widget)
        splitter.setSizes([220, 850])

        footer = QHBoxLayout()
        note = QLabel("All lengths are measured from the tool tip. Collision checks use the tool's outer envelope.")
        note.setWordWrap(True)
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
        # Enter never presses New / Remove / Done; only the picker's OK is a default button.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        if self.picker:
            self.ok_button.setDefault(True)
        if getattr(self, "tabs", None) is not None:
            self._tab_index = self.tabs.currentIndex()
            self.tabs.currentChanged.connect(self._tab_changed)

        # Live preview: any edit refreshes the 3D cutter after a short pause.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(lambda: self._update_preview(silent=True))
        for field in self.fields.values():
            field.valueChanged.connect(self._preview_timer.start)
        self.tool_type.currentTextChanged.connect(self._preview_timer.start)
        self.profile_table.itemChanged.connect(self._preview_timer.start)

    # ------------------------------------------------------------ leaving with unsaved edits (v3 Block 2)
    def _machines_dirty_changed(self, dirty: bool) -> None:
        if getattr(self, "tabs", None) is not None and self.machines_tab is not None:
            index = self.tabs.indexOf(self.machines_tab)
            self.tabs.setTabText(index, "Machines *" if dirty else "Machines")

    def _tab_changed(self, index: int) -> None:
        previous = getattr(self, "_tab_index", index)
        if self.machines_tab is not None and previous == self.tabs.indexOf(self.machines_tab) and index != previous:
            if not self.machines_tab.maybe_save():
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(previous)
                self.tabs.blockSignals(False)
                return
        self._tab_index = index

    def confirm_leave(self) -> bool:
        """Unsaved edits anywhere in the window ask Save / Discard / Cancel before it closes."""
        if self.machines_tab is not None and not self.machines_tab.maybe_save():
            return False
        if not self.picker and not self.maybe_save_tool():
            return False
        return True

    # ------------------------------------------------------------ unsaved tool edits (v3 Block 3)
    def _form_state(self) -> tuple:
        rows = []
        for row in range(self.profile_table.rowCount()):
            cells = [self.profile_table.item(row, col) for col in range(2)]
            rows.append(tuple(c.text().strip() if c is not None else "" for c in cells))
        values = tuple(round(float(field.value()), 6) for field in self.fields.values())
        return (self.name.text().strip(), self.tool_type.currentText(), values, tuple(rows))

    def _set_baseline(self) -> None:
        self._baseline = self._form_state()

    def tool_form_dirty(self) -> bool:
        return self._baseline is not None and self._form_state() != self._baseline

    def maybe_save_tool(self) -> bool:
        """Unsaved tool edits: ask Save / Discard / Cancel. True means it is fine to move on."""
        if not self.tool_form_dirty():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved tool",
            f"Save the changes to {self.name.text().strip() or 'this tool'}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save_tool()
        if answer == QMessageBox.StandardButton.Discard:
            if 0 <= self.selected_index < len(self.tools):
                self._select_tool(self.selected_index)
            else:
                self._set_baseline()
            return True
        return False

    def _tool_row_changed(self, index: int) -> None:
        if index == self.selected_index and self._baseline is not None:
            return
        if not self.picker and self.tool_form_dirty():
            target = self.tools[index] if 0 <= index < len(self.tools) else None
            self.inventory.blockSignals(True)
            self.inventory.setCurrentRow(self.selected_index)
            self.inventory.blockSignals(False)
            if not self.maybe_save_tool():
                return
            index = next((i for i, tool in enumerate(self.tools) if tool is target), -1)
            self.inventory.blockSignals(True)
            self.inventory.setCurrentRow(index)
            self.inventory.blockSignals(False)
        self._select_tool(index)

    def _notify_change(self) -> bool:
        if self.on_change is None:
            return True
        return self.on_change(deepcopy(self.tools)) is not False

    def accept(self) -> None:
        if self.confirm_leave():
            super().accept()

    def reject(self) -> None:  # Esc, the title-bar X and Cancel all come here
        if self.confirm_leave():
            super().reject()

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
        self._load_fields(tool)
        self._update_preview()
        self._set_baseline()

    def _new_tool(self) -> None:
        if not self.maybe_save_tool():
            return
        tool_type = self.tool_type.currentText()
        tool = ToolDefinition(f"New {tool_type}", tool_type)
        tool = replace(tool, flute_length_mm=tool.cutting_length_mm)  # flute follows the cutting length
        self.name.setText(tool.name)
        self.selected_index = -1
        self.inventory.blockSignals(True)
        self.inventory.setCurrentRow(-1)
        self.inventory.blockSignals(False)
        self._load_fields(tool)
        self._update_preview()
        self._baseline = ("", "", (), ())  # a new tool counts as unsaved until Add / update inventory

    # ------------------------------------------------------------ fields (v3 Block 4)
    def _scale(self, key: str) -> float:
        return 25.4 if (self.inches and key in self._length_keys) else 1.0

    def _value_mm(self, key: str) -> float:
        return float(self.fields[key].value()) * self._scale(key)

    def _load_fields(self, tool: ToolDefinition) -> None:
        self._syncing = True
        try:
            self.tool_type.setCurrentText(tool.tool_type)
            for key, field in self.fields.items():
                value = getattr(tool, key)
                field.setValue(int(value) if key == "flute_count" else float(value) / self._scale(key))
            self._flute_linked = abs(tool.flute_length_mm - tool.cutting_length_mm) < 1e-6
            self._profile_mode = tool.profile_mode
            self._set_profile_table(tool.profile_points)
        finally:
            self._syncing = False
        self._apply_type_visibility()

    def _cutting_length_changed(self, value: float) -> None:
        if self._syncing:
            return
        flute = self.fields["flute_length_mm"]
        if self._flute_linked or flute.value() < value:  # follows until edited; never shorter
            self._syncing = True
            flute.setValue(value)
            self._syncing = False

    def _flute_length_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._flute_linked = abs(value - self.fields["cutting_length_mm"].value()) < 1e-9

    TYPE_FIELDS = {
        "End mill": {"corner_radius_mm", "taper_angle_deg", "neck_diameter_mm", "reach_mm", "helix_angle_deg"},
        "Ball end mill": {"taper_angle_deg", "neck_diameter_mm", "reach_mm", "helix_angle_deg"},
        "Router bit": {"corner_radius_mm", "neck_diameter_mm", "reach_mm", "helix_angle_deg"},
        "Spiral flute cutter": {"corner_radius_mm", "neck_diameter_mm", "reach_mm", "helix_angle_deg"},
        "Drill": {"included_angle_deg", "helix_angle_deg"},
        "Tap": {"tip_diameter_mm", "helix_angle_deg"},
        "Reamer": {"tip_diameter_mm", "helix_angle_deg"},
        "Countersink": {"included_angle_deg", "tip_diameter_mm"},
        "V-bit": {"included_angle_deg", "tip_diameter_mm"},
        "Chamfer mill": {"included_angle_deg", "tip_diameter_mm"},
        "Groove cutter": {"neck_diameter_mm"},
        "T-slot cutter": {"neck_diameter_mm"},
        "Dovetail cutter": {"included_angle_deg", "neck_diameter_mm"},
        "Fly cutter": set(),
    }
    COMMON_FIELDS = {"diameter_mm", "shank_diameter_mm", "cutting_length_mm", "flute_length_mm",
                     "overall_length_mm", "flute_count"}

    TYPE_ANGLES = {"Drill": 118.0, "V-bit": 60.0, "Chamfer mill": 90.0, "Countersink": 90.0, "Dovetail cutter": 60.0}

    def _apply_type_visibility(self, *_args) -> None:
        kind = self.tool_type.currentText()
        if not self._syncing and _args and kind in self.TYPE_ANGLES:
            # the owner picked a new type: start from that type's usual angle
            self.fields["included_angle_deg"].setValue(self.TYPE_ANGLES[kind])
        custom = kind == "Custom" or getattr(self, "_profile_mode", "generated") == "custom"
        visible = set(self.fields) if kind == "Custom" else self.COMMON_FIELDS | self.TYPE_FIELDS.get(kind, set())
        for key, field in self.fields.items():
            self.dimensions_form.setRowVisible(field, key in visible)
        head = kind in {"Groove cutter", "T-slot cutter", "Dovetail cutter", "Fly cutter"}
        unit = "in" if self.inches else "mm"
        label = self.dimensions_form.labelForField(self.fields["cutting_length_mm"])
        if label is not None:
            label.setText(f"Head thickness ({unit})" if head else f"Cutting length ({unit})")
        angle_label = self.dimensions_form.labelForField(self.fields["included_angle_deg"])
        if angle_label is not None:
            angle_label.setText({"Drill": "Point angle (°)", "Dovetail cutter": "Dovetail angle (°)"}.get(
                kind, "Included angle (°)"))
        self.profile_widget.setVisible(custom)
        self.custom_button.setVisible(not custom)

    def _edit_as_custom(self) -> None:
        self._profile_mode = "custom"
        try:
            self._set_profile_table(self._make_tool().profile_points)
        except ValueError:
            pass
        self._apply_type_visibility()
        self._update_preview(silent=True)

    def _make_tool(self) -> ToolDefinition:
        values = {}
        for key, field in self.fields.items():
            values[key] = int(field.value()) if key == "flute_count" else self._value_mm(key)
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
        name = self.name.text().strip() or f"Custom {self.tool_type.currentText()}"
        mode = "custom" if (self.tool_type.currentText() == "Custom" or getattr(self, "_profile_mode", "generated") == "custom") else "generated"
        if mode == "generated":
            points = []  # regenerated from the dimensions
        base = self.selected_tool()
        if base is not None:  # keep fields the form does not show (holder, lathe insert...)
            return replace(base, name=name, tool_type=self.tool_type.currentText(), profile_points=points,
                           profile_mode=mode, **values)
        return ToolDefinition(
            name=name,
            tool_type=self.tool_type.currentText(),
            profile_points=points,
            profile_mode=mode,
            **values,
        )

    def _set_profile_table(self, points: list[tuple[float, float]]) -> None:
        self.profile_table.setRowCount(len(points))
        for row, (radius, height) in enumerate(points):
            self.profile_table.setItem(row, 0, QTableWidgetItem(f"{radius:g}"))
            self.profile_table.setItem(row, 1, QTableWidgetItem(f"{height:g}"))

    def _reset_profile(self) -> None:
        self._profile_mode = "generated"
        try:
            tool = self._make_tool()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid profile", str(error))
            return
        self._set_profile_table(default_profile(tool))
        self._apply_type_visibility()
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
            tool = self._make_tool()
            mesh = tool_preview_mesh(tool)
        except (ValueError, TypeError, AttributeError) as error:
            if not silent:
                QMessageBox.warning(self, "Invalid tool profile", str(error))
            return
        self.preview_mesh = mesh
        self.preview.clear_actors()
        self.preview.add_axes(line_width=2)
        colors = [theme_color("toolpath_cut"), theme_color("tool"), theme_color("face")]
        self.preview.add_mesh(mesh, scalars="zone", cmap=colors, clim=[0, 2], n_colors=3,
                              show_scalar_bar=False, smooth_shading=True)
        self.dimension_labels = []
        if self.show_dimensions.isChecked():
            self._draw_dimensions(tool)
        self.preview.view_xz()
        self.preview.reset_camera()
        self.preview.render()

    def _fmt(self, mm: float) -> str:
        return f"{mm / 25.4:.4g} in" if self.inches else f"{mm:.4g} mm"

    def _draw_dimensions(self, tool: ToolDefinition) -> None:
        """Dimension lines beside the tool: each length from the tip, each diameter where it is measured."""
        import pyvista as pv

        right = tool.max_radius_mm() * 1.4 + 1.0
        step = max(tool.max_radius_mm() * 0.45, 1.2)
        lengths = [("Cutting length", tool.cutting_length_mm), ("Flute length", tool.flute_length_mm)]
        if tool.reach_mm > 0:
            lengths.append(("Reach", tool.reach_mm))
        lengths.append(("Overall length", tool.overall_length_mm))
        points, labels = [], []
        color = theme_color("muted")
        for i, (name, length) in enumerate(lengths):
            x = right + i * step
            self.preview.add_mesh(pv.Line((x, 0, 0), (x, 0, length)), color=color, line_width=2)
            for z in (0.0, length):  # end ticks
                self.preview.add_mesh(pv.Line((x - step * 0.2, 0, z), (x + step * 0.2, 0, z)), color=color, line_width=2)
            points.append((x, 0, length))
            labels.append(f"{name} {self._fmt(length)}")
        left = -tool.max_radius_mm() * 1.4 - 1.0
        diameters = [("Cutting Ø", tool.diameter_mm, tool.diameter_mm / 2.0, min(tool.cutting_length_mm, tool.flute_length_mm) * 0.5)]
        shank_z = (max(tool.flute_length_mm, tool.reach_mm) + tool.overall_length_mm) / 2.0
        diameters.append(("Shank Ø", tool.shank_diameter_mm, tool.shank_diameter_mm / 2.0, shank_z))
        if tool.neck_diameter_mm > 0 and tool.reach_mm > tool.flute_length_mm:
            diameters.append(("Neck Ø", tool.neck_diameter_mm, tool.neck_diameter_mm / 2.0,
                              (tool.flute_length_mm + tool.reach_mm) / 2.0))
        for name, size, radius, z in diameters:
            self.preview.add_mesh(pv.Line((-radius, 0, z), (radius, 0, z)), color=color, line_width=2)
            points.append((left, 0, z))
            labels.append(f"{name} {self._fmt(size)}")
        self.dimension_labels = labels
        self.preview.add_point_labels(points, labels, font_size=11, text_color=theme_color("text"),
                                      shape=None, show_points=False, always_visible=True)

    def _save_tool(self) -> bool:
        try:
            tool = self._make_tool()
            create_tool_mesh(tool)
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "Invalid tool", str(error))
            return False
        problems = validate_tool(tool)
        if problems:
            QMessageBox.warning(self, "Invalid tool", "\n".join(problems))
            return False
        if self.selected_index < 0:
            self.tools.append(tool)
            self.selected_index = len(self.tools) - 1
        else:
            self.tools[self.selected_index] = tool
        self._refresh_list()
        self._update_preview()
        self._set_baseline()
        return self._notify_change()

    def _remove_tool(self) -> None:
        if not 0 <= self.selected_index < len(self.tools):
            return
        name = self.tools[self.selected_index].name
        answer = QMessageBox.question(
            self, "Remove tool", f"Remove {name} from the inventory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._baseline = None
        self.tools.pop(self.selected_index)
        self.selected_index = min(self.selected_index, len(self.tools) - 1)
        self._refresh_list()
        if self.tools:
            self._select_tool(self.selected_index)
        else:
            self.preview.clear_actors()
            self.preview.add_axes(line_width=2)
            self.preview.render()
        self._notify_change()
