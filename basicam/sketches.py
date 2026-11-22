from dataclasses import dataclass

import numpy as np
import pyvista as pv


@dataclass
class SketchEntity:
    kind: str
    points: np.ndarray
    construction: bool = False

    def to_mesh(self) -> pv.PolyData:
        if self.kind == "Point":
            return pv.PolyData(self.points, verts=np.array([1, 0], dtype=np.int64))
        line = pv.lines_from_points(self.points)
        line.cell_data["construction"] = np.array([self.construction], dtype=np.uint8)
        return line


def construction_line(first: np.ndarray, second: np.ndarray, extent: float) -> np.ndarray:
    direction = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 1e-12:
        raise ValueError("Construction line points must be distinct.")
    direction /= magnitude
    center = (np.asarray(first, dtype=float) + np.asarray(second, dtype=float)) / 2
    half_length = max(float(extent), magnitude) / 2
    return np.vstack((center - direction * half_length, center + direction * half_length))
