"""Dialogs for part materials, stock kinds and the material library (Materials tab)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from basicam.materials import FAMILIES, PROCESSES, Material, MaterialStore
from basicam.stock import BAR_SHAPES, STOCK_KINDS, StockSpec


def _spin(value: float, maximum: float = 100000.0, decimals: int = 3, suffix: str = " mm") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, maximum)
    spin.setDecimals(decimals)
    spin.setValue(float(value))
    spin.setSuffix(suffix)
    return spin


class SharedMaterialDialog(QDialog):
    """Step 1 after opening several parts: do they all share one material and thickness?"""

    def __init__(self, part_names: list[str], materials: list[str], default_material: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parts material")
        self.same = True
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(part_names)} parts opened. Are all parts the same material type and thickness?"))
        form = QFormLayout()
        self.material = QComboBox()
        self.material.addItems(materials)
        if default_material in materials:
            self.material.setCurrentText(default_material)
        self.thickness = _spin(0.0)
        self.thickness.setSpecialValueText("From part")
        form.addRow("Material", self.material)
        form.addRow("Thickness", self.thickness)
        layout.addLayout(form)
        buttons = QDialogButtonBox()
        self.yes_button = buttons.addButton("Yes, all the same", QDialogButtonBox.ButtonRole.AcceptRole)
        self.no_button = buttons.addButton("No, set each part", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.yes_button.clicked.connect(lambda: self._finish(True))
        self.no_button.clicked.connect(lambda: self._finish(False))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _finish(self, same: bool) -> None:
        self.same = same
        self.accept()


class PartMaterialsDialog(QDialog):
    """One row per part: material, thickness, quantity, allow rotation (nesting)."""

    COLUMNS = ("Part", "Material", "Thickness", "Quantity", "Allow rotation")

    def __init__(self, part_names: list[str], materials: list[str], default_material: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Material for each part")
        self.resize(720, 360)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(part_names), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.rows: list[tuple[QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox]] = []
        for row, name in enumerate(part_names):
            self.table.setCellWidget(row, 0, QLabel(name))
            material = QComboBox()
            material.addItems(materials)
            if default_material in materials:
                material.setCurrentText(default_material)
            thickness = _spin(0.0)
            thickness.setSpecialValueText("From part")
            quantity = QSpinBox()
            quantity.setRange(1, 100000)
            rotation = QCheckBox()
            rotation.setToolTip("Let nesting rotate this part to fit more parts on the stock.")
            for column, widget in enumerate((material, thickness, quantity, rotation), start=1):
                self.table.setCellWidget(row, column, widget)
            self.rows.append((material, thickness, quantity, rotation))
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> list[dict]:
        return [
            {
                "material": m.currentText(),
                "thickness_mm": float(t.value()),
                "quantity": int(q.value()),
                "allow_rotation": bool(r.isChecked()),
            }
            for m, t, q, r in self.rows
        ]


class StockDialog(QDialog):
    """Stock kind and its fields: block, sheet / plate, bar, tube."""

    LABELS = {"block": "Block", "sheet": "Sheet", "plate": "Plate", "bar": "Bar", "tube": "Tube"}

    def __init__(self, spec: StockSpec, materials: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stock")
        layout = QVBoxLayout(self)
        top = QFormLayout()
        self.kind = QComboBox()
        for kind in STOCK_KINDS:
            self.kind.addItem(self.LABELS[kind], kind)
        self.kind.setCurrentIndex(max(0, STOCK_KINDS.index(spec.kind) if spec.kind in STOCK_KINDS else 0))
        self.material = QComboBox()
        self.material.addItems(materials)
        if spec.material in materials:
            self.material.setCurrentText(spec.material)
        top.addRow("Type", self.kind)
        top.addRow("Material", self.material)
        layout.addLayout(top)

        self.pages = QStackedWidget()
        # Block / sheet / plate
        flat = QWidget()
        flat_form = QFormLayout(flat)
        self.length = _spin(spec.length)
        self.width = _spin(spec.width)
        self.thickness = _spin(spec.thickness)
        self.margin = _spin(spec.margin)
        self.margin.setToolTip("Unusable border around the sheet edge; nesting stays inside it.")
        self.grain = QComboBox()
        self.grain.addItems(["None", "Along X", "Along Y"])
        self.grain.setCurrentIndex({"none": 0, "x": 1, "y": 2}.get(spec.grain, 0))
        flat_form.addRow("Length (X)", self.length)
        flat_form.addRow("Width (Y)", self.width)
        flat_form.addRow("Thickness (Z)", self.thickness)
        self.margin_label = QLabel("Edge margin")
        flat_form.addRow(self.margin_label, self.margin)
        self.grain_label = QLabel("Grain")
        flat_form.addRow(self.grain_label, self.grain)
        # Bar / tube
        round_page = QWidget()
        round_form = QFormLayout(round_page)
        self.bar_length = _spin(spec.length)
        self.bar_shape = QComboBox()
        self.bar_shape.addItems([s.capitalize() for s in BAR_SHAPES])
        self.bar_shape.setCurrentIndex(BAR_SHAPES.index(spec.bar_shape) if spec.bar_shape in BAR_SHAPES else 0)
        self.diameter = _spin(spec.diameter)
        self.diameter.setToolTip("Round: diameter. Hex and square: across flats. Tube: outer diameter.")
        self.wall = _spin(spec.wall)
        round_form.addRow("Length (X)", self.bar_length)
        self.shape_label = QLabel("Shape")
        round_form.addRow(self.shape_label, self.bar_shape)
        round_form.addRow("Diameter", self.diameter)
        self.wall_label = QLabel("Wall")
        round_form.addRow(self.wall_label, self.wall)
        self.pages.addWidget(flat)
        self.pages.addWidget(round_page)
        layout.addWidget(self.pages)
        self.error = QLabel("")
        self.error.setObjectName("errorText")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._base = spec
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self._kind_changed()

    def _kind_changed(self) -> None:
        kind = self.kind.currentData()
        round_kind = kind in ("bar", "tube")
        self.pages.setCurrentIndex(1 if round_kind else 0)
        flat = kind in ("sheet", "plate")
        for widget in (self.margin, self.margin_label, self.grain, self.grain_label):
            widget.setVisible(flat)
        for widget in (self.bar_shape, self.shape_label):
            widget.setVisible(kind == "bar")
        for widget in (self.wall, self.wall_label):
            widget.setVisible(kind == "tube")

    def spec(self) -> StockSpec:
        kind = self.kind.currentData()
        round_kind = kind in ("bar", "tube")
        return StockSpec(
            kind=kind,
            length=float(self.bar_length.value() if round_kind else self.length.value()),
            width=float(self.width.value()),
            thickness=float(self.thickness.value()),
            material=self.material.currentText(),
            grain=("none", "x", "y")[self.grain.currentIndex()],
            margin=float(self.margin.value()),
            bar_shape=BAR_SHAPES[self.bar_shape.currentIndex()],
            diameter=float(self.diameter.value()),
            wall=float(self.wall.value()),
            origin_x=self._base.origin_x,
            origin_y=self._base.origin_y,
            origin_z=self._base.origin_z,
            top_at_z0=self._base.top_at_z0,
        )

    def _accept(self) -> None:
        problems = self.spec().validate()
        if problems:
            self.error.setText(" ".join(problems))
            return
        self.accept()


class MaterialsTab(QWidget):
    """Material library editor (a tab in the tool inventory window)."""

    def __init__(self, store: MaterialStore, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.materials = store.list()
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list = QListWidget()
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete)
        row.addWidget(add)
        row.addWidget(delete)
        left.addLayout(row)
        root.addLayout(left, 1)
        form = QFormLayout()
        self.name = QLineEdit()
        self.family = QComboBox()
        self.family.addItems(list(FAMILIES))
        self.density = _spin(1.0, 30.0, 3, " g/cm³")
        self.conductive = QCheckBox("Electrically conductive")
        self.conductive.setToolTip("Plasma and wire EDM can only cut conductive materials.")
        form.addRow("Name", self.name)
        form.addRow("Family", self.family)
        form.addRow("Density", self.density)
        form.addRow("", self.conductive)
        self.process_checks: dict[str, QCheckBox] = {}
        for process in PROCESSES:
            check = QCheckBox(process.replace("_", " ").capitalize())
            self.process_checks[process] = check
            form.addRow("Can be cut by" if process == PROCESSES[0] else "", check)
        save = QPushButton("Save material")
        save.clicked.connect(self._save_current)
        form.addRow(save)
        root.addLayout(form, 2)
        self.list.currentRowChanged.connect(self._show)
        self._refresh()

    def _refresh(self, select: int = 0) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems([m.name for m in self.materials])
        self.list.blockSignals(False)
        if self.materials:
            self.list.setCurrentRow(min(select, len(self.materials) - 1))
            self._show(self.list.currentRow())

    def _show(self, row: int) -> None:
        if not (0 <= row < len(self.materials)):
            return
        material = self.materials[row]
        self.name.setText(material.name)
        self.family.setCurrentText(material.family)
        self.density.setValue(material.density_g_cm3)
        self.conductive.setChecked(material.conductive)
        for process, check in self.process_checks.items():
            check.setChecked(bool(material.processes.get(process, True)))

    def _save_current(self) -> None:
        row = self.list.currentRow()
        name = self.name.text().strip()
        if not name:
            return
        material = Material(
            name,
            self.family.currentText(),
            float(self.density.value()),
            {p: c.isChecked() for p, c in self.process_checks.items()},
            self.conductive.isChecked(),
        )
        if 0 <= row < len(self.materials):
            self.materials[row] = material
        else:
            self.materials.append(material)
            row = len(self.materials) - 1
        self.store.save_all(self.materials)
        self._refresh(row)

    def _add(self) -> None:
        base = "New material"
        name, n = base, 2
        while any(m.name == name for m in self.materials):
            name, n = f"{base} {n}", n + 1
        self.materials.append(Material(name))
        self.store.save_all(self.materials)
        self._refresh(len(self.materials) - 1)

    def _delete(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.materials) and len(self.materials) > 1:
            del self.materials[row]
            self.store.save_all(self.materials)
            self._refresh(max(0, row - 1))


__all__ = ["MaterialsTab", "PartMaterialsDialog", "SharedMaterialDialog", "StockDialog"]
