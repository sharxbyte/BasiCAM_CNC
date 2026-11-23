from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from basicam.models import PartModel, PartRole
from basicam.operations import ENTRY_FIELDS, EXTRA_FIELDS, Operation
from basicam.setups import MachiningSetup
from basicam.sketch_workflow import ParametricSketchFeature, SketchDimension, SketchWorkflowManager


BCAM_FORMAT_VERSION = 1


@dataclass
class WorkspaceSnapshot:
    """Decoupled data representation of workspace state (independent of PyVista/VTK view objects)."""

    job_name: str = "Untitled job"
    job_notes: str = ""
    stock_material: str = "Unspecified"
    stock_dimensions: dict[str, float] = field(default_factory=lambda: {"X": 100.0, "Y": 80.0, "Z": 20.0})
    stock_note: str = ""
    stock_imported: bool = False
    stock_part_index: int | None = None
    active_setup_index: int = 0
    setups: list[MachiningSetup] = field(default_factory=list)
    sketch_features: list[ParametricSketchFeature] = field(default_factory=list)
    active_sketch_id: str | None = None
    parts_meta: list[dict[str, Any]] = field(default_factory=list)
    part_meshes: list[pv.PolyData] = field(default_factory=list)
    part_edge_meshes: list[pv.PolyData | None] = field(default_factory=list)
    part_breps: list[bytes | None] = field(default_factory=list)
    part_solids: list[bytes | None] = field(default_factory=list)  # only when solid differs from B-rep
    stock_mesh: pv.PolyData | None = None
    form_draft: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    extras: dict[str, Any] = field(default_factory=dict)


def shape_to_brep_bytes(shape: Any) -> bytes | None:
    """Serialize an Open CASCADE shape to BREP bytes (lossless, keeps exact faces/edges)."""
    if shape is None:
        return None
    try:
        from OCP.BRepTools import BRepTools

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "shape.brep")
            if not BRepTools.Write_s(shape, path):
                return None
            with open(path, "rb") as handle:
                return handle.read()
    except Exception:
        return None


def brep_bytes_to_shape(blob: bytes | None) -> Any | None:
    if not blob:
        return None
    try:
        from OCP.BRep import BRep_Builder
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Shape

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "shape.brep")
            with open(path, "wb") as handle:
                handle.write(blob)
            shape = TopoDS_Shape()
            if not BRepTools.Read_s(shape, path, BRep_Builder()):
                return None
            return shape
    except Exception:
        return None


def _serialize_sketch_feature(feat: ParametricSketchFeature) -> dict[str, Any]:
    return {
        "feature_id": feat.feature_id,
        "name": feat.name,
        "kind": feat.kind,
        "plane": feat.plane,
        "points": np.asarray(feat.points, dtype=float).tolist(),
        "projected_points": (
            np.asarray(feat.projected_points, dtype=float).tolist()
            if feat.projected_points is not None
            else None
        ),
        "is_projected": bool(feat.is_projected),
        "construction": bool(feat.construction),
        "parent_feature_id": feat.parent_feature_id,
        "constraints": list(feat.constraints),
        "defined": bool(feat.defined),
        "has_error": bool(feat.has_error),
        "error_message": feat.error_message,
        "assigned_operation_name": feat.assigned_operation_name,
        "assigned_setup_name": feat.assigned_setup_name,
        "assigned_setup_color": feat.assigned_setup_color,
        "dimensions": [asdict(d) for d in feat.dimensions],
        "entity_id": feat.feature_id,
        "owner_sketch_id": feat.owner_sketch_id,
        "variant": feat.variant,
        "state": feat.state,
        "broken_reason": feat.broken_reason,
        "status": feat.status,
        "control_points": (
            np.asarray(feat.control_points, dtype=float).tolist()
            if getattr(feat, "control_points", None) is not None else None
        ),
    }


def _deserialize_sketch_feature(data: dict[str, Any]) -> ParametricSketchFeature:
    dims = [
        SketchDimension(
            dim_id=str(d.get("dim_id", "")),
            name=str(d.get("name", "")),
            kind=str(d.get("kind", "Length")),
            value=float(d.get("value", 0.0)),
            tolerance=float(d.get("tolerance", 0.0)),
            valid=bool(d.get("valid", True)),
            error_message=str(d.get("error_message", "")),
            target_ids=[str(t) for t in d.get("target_ids", [])],
            driving=bool(d.get("driving", True)),
            unit_text=str(d.get("unit_text", "mm")),
            sig_figs=int(d.get("sig_figs", 4)),
            point_refs=[int(i) for i in d.get("point_refs", []) or []],
            label_offset=[float(v) for v in d.get("label_offset", []) or []],
        )
        for d in data.get("dimensions", [])
    ]
    proj_raw = data.get("projected_points")
    proj_arr = np.asarray(proj_raw, dtype=float) if proj_raw is not None else None
    return ParametricSketchFeature(
        feature_id=str(data.get("feature_id", "sk_000")),
        name=str(data.get("name", "Sketch")),
        kind=str(data.get("kind", "Line")),
        plane=str(data.get("plane", "XY plane")),
        points=np.asarray(data.get("points", []), dtype=float).reshape(-1, 3),
        projected_points=proj_arr,
        is_projected=bool(data.get("is_projected", False)),
        construction=bool(data.get("construction", False)),
        parent_feature_id=data.get("parent_feature_id"),
        constraints=list(data.get("constraints", [])),
        defined=bool(data.get("defined", False)),
        has_error=bool(data.get("has_error", False)),
        error_message=str(data.get("error_message", "")),
        assigned_operation_name=data.get("assigned_operation_name"),
        assigned_setup_name=data.get("assigned_setup_name"),
        assigned_setup_color=data.get("assigned_setup_color"),
        dimensions=dims,
        owner_sketch_id=str(data.get("owner_sketch_id", "")),
        variant=str(data.get("variant", "")),
        state=str(data.get("state", "underdefined")),
        broken_reason=str(data.get("broken_reason", "")),
        status=str(data.get("status", "")),
        control_points=(
            np.asarray(data["control_points"], dtype=float).reshape(-1, 3)
            if data.get("control_points") is not None else None
        ),
    )


def _serialize_setups(setups: list[MachiningSetup]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for setup in setups:
        out.append(
            {
                "name": setup.name,
                "orientation": setup.orientation,
                "origin_mm": list(setup.origin_mm),
                "color": setup.color,
                "operations": [
                    {
                        **asdict(op),
                        "overlap_pct": getattr(op, "overlap_pct", None),
                        "finish_quality": getattr(op, "finish_quality", None),
                        "surface": getattr(op, "surface", None),
                        "single_run": getattr(op, "single_run", None),
                    }
                    for op in setup.operations
                ],
                "work_frame": getattr(setup, "work_frame", None),
                "work_offset": getattr(setup, "work_offset", "G54"),
                "safe_z_mm": float(getattr(setup, "safe_z_mm", 5.0)),
                "retract_clearance_mm": float(getattr(setup, "retract_clearance_mm", 2.0)),
                "stock_clearance_mm": float(getattr(setup, "stock_clearance_mm", 1.0)),
                "work_type": getattr(setup, "work_type", ""),
                "work_surface": getattr(setup, "work_surface", None),
                "rotary_index_deg": float(getattr(setup, "rotary_index_deg", 0.0)),
                "tool_orientation": list(setup.tool_orientation) if getattr(setup, "tool_orientation", None) else None,
                "rotary_angles": getattr(setup, "rotary_angles", None),
                "turning": getattr(setup, "turning", None),
            }
        )
    return out


def _with_extras(operation: Operation, raw: dict[str, Any]) -> Operation:
    if raw.get("overlap_pct") is not None:
        operation.overlap_pct = float(raw["overlap_pct"])
    if raw.get("finish_quality"):
        operation.finish_quality = str(raw["finish_quality"])
    for name in ENTRY_FIELDS:  # depth/entry fields; old files keep the dataclass defaults
        if raw.get(name) is not None:
            default = getattr(operation, name)
            setattr(operation, name, type(default)(raw[name]))
    for name in ("surface", "single_run"):
        if raw.get(name) is not None:
            setattr(operation, name, raw[name])
    for name in EXTRA_FIELDS:
        if raw.get(name) is not None and hasattr(operation, name):
            default = getattr(operation, name)
            value = raw[name]
            setattr(operation, name, type(default)(value) if isinstance(default, (int, float, str, bool)) else value)
    return operation


def _deserialize_setups(raw_setups: list[dict[str, Any]]) -> list[MachiningSetup]:
    out: list[MachiningSetup] = []
    for s in raw_setups:
        ops = [
            _with_extras(
            Operation(
                name=str(op.get("name", "01 · Rough")),
                kind=str(op.get("kind", "Rough")),
                tool=str(op.get("tool", "End mill")),
                rpm=int(op.get("rpm", 6000)),
                feed_mm_min=float(op.get("feed_mm_min", 450.0)),
                stepdown_mm=float(op.get("stepdown_mm", 2.0)),
                depth_mm=float(op.get("depth_mm", 8.0)),
                direction=str(op.get("direction", "climb")),
                tool_diameter_mm=float(op.get("tool_diameter_mm", 6.0)),
                target=str(op.get("target", "Whole part")),
            ),
            op,
            )
            for op in s.get("operations", [])
        ]
        orig_raw = s.get("origin_mm", [0.0, 0.0, 0.0])
        origin_tuple = (
            float(orig_raw[0] if len(orig_raw) > 0 else 0.0),
            float(orig_raw[1] if len(orig_raw) > 1 else 0.0),
            float(orig_raw[2] if len(orig_raw) > 2 else 0.0),
        )
        out.append(
            MachiningSetup(
                name=str(s.get("name", "Setup 1")),
                orientation=str(s.get("orientation", "Top (+Z)")),
                origin_mm=origin_tuple,
                color=str(s.get("color", "#c2df54")),
                operations=ops,
                work_frame=s.get("work_frame") or None,
                work_offset=str(s.get("work_offset", "G54")),
                safe_z_mm=float(s.get("safe_z_mm", 5.0)),
                retract_clearance_mm=float(s.get("retract_clearance_mm", 2.0)),
                stock_clearance_mm=float(s.get("stock_clearance_mm", 1.0)),
                work_type=str(s.get("work_type", "") or ""),
                work_surface=s.get("work_surface"),
                rotary_index_deg=float(s.get("rotary_index_deg", 0.0) or 0.0),
                tool_orientation=tuple(float(v) for v in s["tool_orientation"]) if s.get("tool_orientation") else None,
                rotary_angles=s.get("rotary_angles"),
                turning=s.get("turning"),
            )
        )
    return out


_VTP_MAGIC = b"BCVTP1\n"


def _mesh_to_lossless_bytes(mesh: pv.PolyData) -> bytes:
    """Serialize a PolyData exactly (points, verts/lines/polys, point and cell arrays) as binary VTP.

    Uses VTK's own writer: it works across PyVista versions (PolyData.cells no longer exists).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mesh.vtp")
        mesh.save(path, binary=True)
        with open(path, "rb") as handle:
            return _VTP_MAGIC + handle.read()


def _legacy_mesh_to_bytes(mesh: pv.PolyData) -> bytes:
    """Old payload format (kept only so the reader below stays testable)."""
    pts = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    cell_data_payload: dict[str, list[Any]] = {}
    for k, arr in mesh.cell_data.items():
        np_arr = np.asarray(arr)
        cell_data_payload[str(k)] = {
            "dtype": str(np_arr.dtype),
            "shape": list(np_arr.shape),
            "data": np_arr.tolist(),
        }
    header = {
        "n_points": int(pts.shape[0]),
        "n_cells_arr": int(cells.shape[0]),
        "cell_data": cell_data_payload,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    len_prefix = int(len(header_bytes)).to_bytes(4, "little")
    return len_prefix + header_bytes + pts.tobytes() + cells.tobytes()


def _mesh_from_lossless_bytes(blob: bytes) -> pv.PolyData:
    if blob.startswith(_VTP_MAGIC):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mesh.vtp")
            with open(path, "wb") as handle:
                handle.write(blob[len(_VTP_MAGIC):])
            return pv.read(path)
    if len(blob) < 4:
        return pv.PolyData()
    hdr_len = int.from_bytes(blob[:4], "little")
    hdr_end = 4 + hdr_len
    header = json.loads(blob[4:hdr_end].decode("utf-8"))
    n_pts = int(header.get("n_points", 0))
    n_cells_arr = int(header.get("n_cells_arr", 0))
    body = memoryview(blob)[hdr_end:]
    pts_nbytes = n_pts * 3 * 8
    pts = np.frombuffer(body[:pts_nbytes], dtype=np.float64).reshape((n_pts, 3)).copy()
    cells = np.frombuffer(body[pts_nbytes : pts_nbytes + n_cells_arr * 8], dtype=np.int64).copy()
    mesh = pv.PolyData(pts, cells) if n_cells_arr > 0 else pv.PolyData(pts)
    for k, meta in header.get("cell_data", {}).items():
        arr = np.asarray(meta["data"], dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
        mesh.cell_data[str(k)] = arr
    return mesh


def write_snapshot_to_directory(snapshot: WorkspaceSnapshot, target_dir: str | Path) -> Path:
    """Writes uncompressed project files to a working/autosave directory."""
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    parts_entries: list[dict[str, Any]] = []
    for idx, part_info in enumerate(snapshot.parts_meta):
        entry = dict(part_info)
        mesh = snapshot.part_meshes[idx] if idx < len(snapshot.part_meshes) else None
        if mesh is not None and mesh.n_points > 0:
            mesh_file = f"part_{idx}.bin"
            (dest / mesh_file).write_bytes(_mesh_to_lossless_bytes(mesh))
            entry["mesh_file"] = mesh_file
        edge_mesh = snapshot.part_edge_meshes[idx] if idx < len(snapshot.part_edge_meshes) else None
        if edge_mesh is not None and edge_mesh.n_points > 0:
            (dest / f"part_{idx}_edges.bin").write_bytes(_mesh_to_lossless_bytes(edge_mesh))
            entry["edge_file"] = f"part_{idx}_edges.bin"
        brep_blob = snapshot.part_breps[idx] if idx < len(snapshot.part_breps) else None
        if brep_blob:
            (dest / f"part_{idx}.brep").write_bytes(brep_blob)
            entry["brep_file"] = f"part_{idx}.brep"
        solid_blob = snapshot.part_solids[idx] if idx < len(snapshot.part_solids) else None
        if solid_blob:
            (dest / f"part_{idx}_solid.brep").write_bytes(solid_blob)
            entry["solid_file"] = f"part_{idx}_solid.brep"
        parts_entries.append(entry)

    stock_file: str | None = None
    if snapshot.stock_mesh is not None and snapshot.stock_mesh.n_points > 0:
        stock_file = "stock_mesh.bin"
        (dest / stock_file).write_bytes(_mesh_to_lossless_bytes(snapshot.stock_mesh))

    manifest = {
        "format": "BasiCAM_CNC_BCAM",
        "version": BCAM_FORMAT_VERSION,
        "saved_at": snapshot.timestamp,
        "job_name": snapshot.job_name,
        "job_notes": snapshot.job_notes,
        "stock_material": snapshot.stock_material,
        "stock_dimensions": dict(snapshot.stock_dimensions),
        "stock_note": snapshot.stock_note,
        "stock_imported": bool(snapshot.stock_imported),
        "stock_part_index": snapshot.stock_part_index,
        "stock_mesh_file": stock_file,
        "active_setup_index": int(snapshot.active_setup_index),
        "active_sketch_id": snapshot.active_sketch_id,
        "setups": _serialize_setups(snapshot.setups),
        "sketch_features": [_serialize_sketch_feature(f) for f in snapshot.sketch_features],
        "parts": parts_entries,
        "form_draft": dict(snapshot.form_draft),
    }
    manifest_path = dest / "project.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dest / "extras.json").write_text(json.dumps(dict(snapshot.extras or {}), indent=2), encoding="utf-8")
    return manifest_path


def read_snapshot_from_directory(source_dir: str | Path) -> WorkspaceSnapshot:
    src = Path(source_dir)
    manifest_path = src / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_meshes: list[pv.PolyData] = []
    part_edge_meshes: list[pv.PolyData | None] = []
    part_breps: list[bytes | None] = []
    part_solids: list[bytes | None] = []
    parts_meta: list[dict[str, Any]] = []
    for p in manifest.get("parts", []):
        p_copy = dict(p)
        mfile = p_copy.get("mesh_file")
        if mfile and (src / mfile).exists():
            part_meshes.append(_mesh_from_lossless_bytes((src / mfile).read_bytes()))
        else:
            part_meshes.append(pv.PolyData())
        efile = p_copy.get("edge_file")
        part_edge_meshes.append(
            _mesh_from_lossless_bytes((src / efile).read_bytes()) if efile and (src / efile).exists() else None
        )
        bfile = p_copy.get("brep_file")
        part_breps.append((src / bfile).read_bytes() if bfile and (src / bfile).exists() else None)
        sfile = p_copy.get("solid_file")
        part_solids.append((src / sfile).read_bytes() if sfile and (src / sfile).exists() else None)
        parts_meta.append(p_copy)

    stock_mesh: pv.PolyData | None = None
    smfile = manifest.get("stock_mesh_file")
    if smfile and (src / smfile).exists():
        stock_mesh = _mesh_from_lossless_bytes((src / smfile).read_bytes())

    extras: dict[str, Any] = {}
    extras_path = src / "extras.json"
    if extras_path.exists():
        try:
            loaded = json.loads(extras_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                extras = loaded
        except (OSError, ValueError):
            extras = {}

    return WorkspaceSnapshot(
        job_name=str(manifest.get("job_name", "Untitled job")),
        job_notes=str(manifest.get("job_notes", "")),
        stock_material=str(manifest.get("stock_material", "Unspecified")),
        stock_dimensions={
            k: float(v) for k, v in manifest.get("stock_dimensions", {"X": 100.0, "Y": 80.0, "Z": 20.0}).items()
        },
        stock_note=str(manifest.get("stock_note", "")),
        stock_imported=bool(manifest.get("stock_imported", False)),
        stock_part_index=manifest.get("stock_part_index"),
        active_setup_index=int(manifest.get("active_setup_index", 0)),
        setups=_deserialize_setups(manifest.get("setups", [])),
        sketch_features=[_deserialize_sketch_feature(f) for f in manifest.get("sketch_features", [])],
        active_sketch_id=manifest.get("active_sketch_id"),
        parts_meta=parts_meta,
        part_meshes=part_meshes,
        part_edge_meshes=part_edge_meshes,
        part_breps=part_breps,
        part_solids=part_solids,
        stock_mesh=stock_mesh,
        form_draft=dict(manifest.get("form_draft", {})),
        timestamp=float(manifest.get("saved_at", time.time())),
        extras=extras,
    )


def save_bcam_bundle(snapshot: WorkspaceSnapshot, bcam_path: str | Path) -> Path:
    """Writes a self-contained compressed .BCAM zip archive containing project.json and lossless geometry."""
    out_path = Path(bcam_path)
    if out_path.suffix.lower() != ".bcam":
        out_path = out_path.with_suffix(".bcam")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        stage_dir = Path(tmpdir) / "stage"
        write_snapshot_to_directory(snapshot, stage_dir)
        tmp_zip = Path(tmpdir) / "bundle.bcam.tmp"
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for file_path in stage_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(stage_dir).as_posix())
        shutil.move(str(tmp_zip), str(out_path))
    return out_path


def load_bcam_bundle(bcam_path: str | Path) -> WorkspaceSnapshot:
    """Loads a .BCAM bundle archive into a WorkspaceSnapshot."""
    src_path = Path(bcam_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir) / "extracted"
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(extract_dir)
        return read_snapshot_from_directory(extract_dir)


class CrashResilientSessionManager:
    """
    Manages uncompressed working/autosave state ('saving' -> 'last-good') and
    graceful-exit crash detection.
    """

    def __init__(self, cache_root: str | Path | None = None) -> None:
        base = Path(cache_root) if cache_root is not None else (
            Path(os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            / "BasiCAM_CNC"
            / "session_cache"
        )
        self.cache_dir = base
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.saving_dir = self.cache_dir / "saving"
        self.last_good_dir = self.cache_dir / "last-good"
        self.clean_exit_flag = self.cache_dir / "clean_exit.flag"
        self.dirty_changes = 0
        self.last_autosave_time = time.monotonic()

    def mark_session_started(self) -> None:
        """Clears clean-exit marker at startup so an unexpected termination is detected as a crash."""
        if self.clean_exit_flag.exists():
            self.clean_exit_flag.unlink()

    def mark_clean_exit(self) -> None:
        """Records that the user exited gracefully (saved or intentionally discarded)."""
        self.clean_exit_flag.write_text(json.dumps({"exited_at": time.time(), "clean": True}), encoding="utf-8")

    def had_unclean_exit(self) -> bool:
        """Returns True if previous session did not exit cleanly and a recoverable last-good backup exists."""
        if self.clean_exit_flag.exists():
            return False
        return (self.last_good_dir / "project.json").exists() or (self.saving_dir / "project.json").exists()

    def record_change(self, count: int = 1) -> None:
        self.dirty_changes += max(int(count), 1)

    def should_autosave(self, interval_seconds: float = 300.0, change_threshold: int = 10) -> bool:
        if self.dirty_changes <= 0:
            return False
        elapsed = time.monotonic() - self.last_autosave_time
        return self.dirty_changes >= change_threshold or elapsed >= interval_seconds

    def autosave_atomic(self, snapshot: WorkspaceSnapshot) -> Path:
        """
        Writes uncompressed state to 'saving' first, then duplicates cleanly to 'last-good'
        so a power loss mid-write never corrupts 'last-good'.
        """
        if self.saving_dir.exists():
            shutil.rmtree(self.saving_dir, ignore_errors=True)
        write_snapshot_to_directory(snapshot, self.saving_dir)

        tmp_good = self.cache_dir / "last-good.staging"
        if tmp_good.exists():
            shutil.rmtree(tmp_good, ignore_errors=True)
        shutil.copytree(self.saving_dir, tmp_good)

        if self.last_good_dir.exists():
            shutil.rmtree(self.last_good_dir, ignore_errors=True)
        shutil.move(str(tmp_good), str(self.last_good_dir))

        self.dirty_changes = 0
        self.last_autosave_time = time.monotonic()
        return self.last_good_dir / "project.json"

    def recover_last_good_snapshot(self) -> WorkspaceSnapshot | None:
        if (self.last_good_dir / "project.json").exists():
            return read_snapshot_from_directory(self.last_good_dir)
        if (self.saving_dir / "project.json").exists():
            return read_snapshot_from_directory(self.saving_dir)
        return None


@dataclass
class HistoryEntry:
    label: str
    snapshot: WorkspaceSnapshot


class WorkspaceHistoryManager:
    """
    Maintains a 15-step undo/redo history stack and clipboard for workspace edits,
    plus single-press layered deletion (tool -> path -> sketch -> object).
    """

    def __init__(self, max_steps: int = 15) -> None:
        self.max_steps = max_steps
        self._undo_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []
        self.clipboard: dict[str, Any] | None = None

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def push_state(self, label: str, snapshot: WorkspaceSnapshot) -> None:
        self._undo_stack.append(HistoryEntry(label=label, snapshot=copy.deepcopy(snapshot)))
        if len(self._undo_stack) > self.max_steps:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self, current_snapshot: WorkspaceSnapshot) -> tuple[bool, str, WorkspaceSnapshot | None]:
        if not self._undo_stack:
            return False, "Nothing to undo.", None
        self._redo_stack.append(HistoryEntry(label="Redo state", snapshot=copy.deepcopy(current_snapshot)))
        if len(self._redo_stack) > self.max_steps:
            self._redo_stack.pop(0)
        entry = self._undo_stack.pop()
        return True, f"Undid: {entry.label}", entry.snapshot

    def redo(self, current_snapshot: WorkspaceSnapshot) -> tuple[bool, str, WorkspaceSnapshot | None]:
        if not self._redo_stack:
            return False, "Nothing to redo.", None
        self._undo_stack.append(HistoryEntry(label="Undo state", snapshot=copy.deepcopy(current_snapshot)))
        if len(self._undo_stack) > self.max_steps:
            self._undo_stack.pop(0)
        entry = self._redo_stack.pop()
        return True, f"Redid: {entry.label}", entry.snapshot
