from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pyvista as pv


class PartRole(str, Enum):
    STOCK = "Stock"
    TARGET = "Target part"
    FIXTURE = "Fixture / keep-out"


@dataclass
class PartModel:
    name: str
    mesh: pv.PolyData
    source_path: Path | None = None
    imported: bool = False
    brep_shape: object | None = None
    solid_shape: object | None = None
    edge_mesh: pv.PolyData | None = None
    solid_conversion_message: str = ""
    native_brep: bool = False
    role: PartRole = PartRole.TARGET
    note: str = ""
    feature_notes: dict[int, str] = field(default_factory=dict)
    material: str = ""  # "" = use the job's stock material
    thickness_mm: float = 0.0  # stock thickness for this part; 0 = from its geometry
    quantity: int = 1
    allow_rotation: bool = False  # nesting may rotate this part (off by default)

    @property
    def geometry_type(self) -> str:
        if self.solid_shape is not None:
            return "Solid"
        if self.brep_shape is not None:
            return "Surface model"
        return "Mesh / drawing"