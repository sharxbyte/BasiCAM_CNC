"""Export a part or the whole workspace as STL (pyvista) or STEP (Open CASCADE).

Returns the list of items that could not be exported so the user can be told.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv

EXPORT_FORMATS = {".stl": "STL mesh", ".step": "STEP solid", ".stp": "STEP solid"}


def export_parts(parts, path: str | Path) -> tuple[Path, list[str]]:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in EXPORT_FORMATS:
        raise ValueError("Choose a .stl or .step file name.")
    skipped: list[str] = []
    if suffix == ".stl":
        meshes = []
        for part in parts:
            mesh = getattr(part, "mesh", None)
            if mesh is None or mesh.n_cells == 0:
                skipped.append(f"{part.name}: no mesh")
                continue
            surface = mesh.extract_surface().triangulate()
            if surface.n_cells == 0:
                skipped.append(f"{part.name}: linework only (STL needs faces)")
                continue
            meshes.append(surface)
        if not meshes:
            raise ValueError("Nothing to export: no part has faces.")
        merged = meshes[0] if len(meshes) == 1 else pv.merge(meshes)
        merged.save(str(target))
        return target, skipped

    from OCP.BRep import BRep_Builder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    count = 0
    for part in parts:
        shape = getattr(part, "solid_shape", None) or getattr(part, "brep_shape", None)
        if shape is None:
            skipped.append(f"{part.name}: mesh only, no B-rep (export as STL instead)")
            continue
        builder.Add(compound, shape)
        count += 1
    if not count:
        raise ValueError("Nothing to export: no part has B-rep geometry.")
    writer = STEPControl_Writer()
    if writer.Transfer(compound, STEPControl_AsIs) != IFSelect_RetDone or writer.Write(str(target)) != IFSelect_RetDone:
        raise ValueError("Open CASCADE could not write the STEP file.")
    return target, skipped
