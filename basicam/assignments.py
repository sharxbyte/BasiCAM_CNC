"""Tool-assignment rules, memory and the generation gate.

* Tools can be assigned only to fully defined, non-construction sketch entities.
* Each sketch entity is Construction, Future operation, or has a valid tool path.
* The ledger remembers which operations used an entity, by entity id, so that after
  the sketch is modified and fully defined again the user can retain or re-check them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

STATUS_FUTURE = "future"


class SketchStatus(str, Enum):
    CONSTRUCTION = "Construction"
    FUTURE = "Future operation"
    TOOLPATH = "Tool path"
    UNTOOLED = "Untooled"


def entity_operations(feat, setups) -> list[str]:
    targets = {feat.name, feat.feature_id}
    return [op.name for setup in setups for op in setup.operations if op.target in targets]


def sketch_status(feat, setups) -> SketchStatus:
    if feat.construction:
        return SketchStatus.CONSTRUCTION
    if entity_operations(feat, setups):
        return SketchStatus.TOOLPATH
    if getattr(feat, "status", "") == STATUS_FUTURE:
        return SketchStatus.FUTURE
    return SketchStatus.UNTOOLED


def can_assign_tool(feat, sketch_problems: list | None = None) -> tuple[bool, str]:
    """Only fully defined, non-construction, error-free entities can take a tool.

    ``sketch_problems`` (contours.check_sketch of its sketch): gaps, overlaps, zero-length or
    self-intersecting entities also block the assignment, with the reason and its location."""
    for problem in sketch_problems or []:
        x, y, z = (float(c) for c in problem.location)
        return False, f"Check Sketch: {problem.message} (at {x:.3f}, {y:.3f}, {z:.3f} mm)."
    if feat.construction:
        return False, f"{feat.name} is construction geometry and needs no tool."
    if feat.has_error:
        return False, f"{feat.name} could not be solved; fix or revert it first."
    if getattr(feat, "state", "") != "defined":
        return False, f"{feat.name} is under-defined; add dimensions or relations and anchor it first."
    return True, ""


def untooled(features: Iterable, setups) -> list:
    return [f for f in features if sketch_status(f, setups) is SketchStatus.UNTOOLED]


def generation_gate(features: Iterable, setups, *, single_operation: bool = False, manual_control: bool = False) -> tuple[bool, list]:
    """True when G-code may be generated; else the untooled entities that block it."""
    if single_operation or manual_control:
        return True, []
    blocking = untooled(features, setups)
    return not blocking, blocking


def mark_future(features: Iterable) -> int:
    count = 0
    for feat in features:
        feat.status = STATUS_FUTURE
        count += 1
    return count


def geometry_hash(feat) -> str:
    data = np.round(np.asarray(feat.points, dtype=float), 6).tobytes()
    return hashlib.sha1(data + feat.kind.encode()).hexdigest()[:16]


@dataclass
class AssignmentLedger:
    """entity id -> operation names, plus the geometry hash at the last confirmation."""

    operations: dict[str, list[str]] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    pending: set[str] = field(default_factory=set)  # changed while assigned; ask when defined again

    def record(self, feat, setups) -> None:
        ops = entity_operations(feat, setups)
        if ops:
            self.operations[feat.feature_id] = ops
            self.hashes[feat.feature_id] = geometry_hash(feat)

    def observe(self, features: Iterable, setups) -> list[str]:
        """Call after every solve; returns entity ids whose assignments need the retain question now."""
        ask: list[str] = []
        for feat in features:
            fid = feat.feature_id
            if fid not in self.operations:
                self.record(feat, setups)
                continue
            if geometry_hash(feat) != self.hashes.get(fid) or feat.has_error or getattr(feat, "state", "") != "defined":
                self.pending.add(fid)  # modified (geometry, or no longer fully defined)
            if fid in self.pending and getattr(feat, "state", "") == "defined" and not feat.has_error:
                ask.append(fid)
        return ask

    def confirm(self, feat, setups) -> None:
        self.pending.discard(feat.feature_id)
        self.operations[feat.feature_id] = entity_operations(feat, setups) or self.operations.get(feat.feature_id, [])
        self.hashes[feat.feature_id] = geometry_hash(feat)

    def forget(self, entity_id: str) -> None:
        self.pending.discard(entity_id)
        self.operations.pop(entity_id, None)
        self.hashes.pop(entity_id, None)

    def to_dict(self) -> dict:
        return {"operations": self.operations, "hashes": self.hashes, "pending": sorted(self.pending)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "AssignmentLedger":
        data = data or {}
        return cls(
            {str(k): list(v) for k, v in dict(data.get("operations", {})).items()},
            {str(k): str(v) for k, v in dict(data.get("hashes", {})).items()},
            set(data.get("pending", [])),
        )


def check_assignments_walk(feature_ids: Iterable[str], setups) -> list[tuple[int, int]]:
    """Operations to confirm, from the first one assigned to the sketch downward through the stack."""
    flat = [(s, o, op) for s, setup in enumerate(setups) for o, op in enumerate(setup.operations)]
    ids = set(feature_ids)
    first = next((i for i, (_s, _o, op) in enumerate(flat) if op.target in ids or getattr(op, "_target_id", "") in ids), None)
    if first is None:
        return []
    return [(s, o) for s, o, _op in flat[first:]]


class RetainAssignmentsDialog(QDialog):
    """'Sketch changed, would you like to retain the tool assignments?' Yes / No / Check assignments."""

    def __init__(self, sketch_name: str, operations: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sketch changed")
        self.choice = "yes"
        layout = QVBoxLayout(self)
        text = QLabel(
            f"{sketch_name} changed. Would you like to retain the tool assignments?\n\n" + "\n".join(f"• {o}" for o in operations)
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        row = QHBoxLayout()
        for label, key, tip in (
            ("Yes", "yes", "Keep the operations on the changed geometry."),
            ("No", "no", "Remove these operations from the stack."),
            ("Check assignments", "check", "Confirm each operation from the first one for this sketch downward."),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, k=key: self._choose(k))
            row.addWidget(button)
        layout.addLayout(row)

    def _choose(self, key: str) -> None:
        self.choice = key
        self.accept()
