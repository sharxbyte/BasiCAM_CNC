"""List third-party packages imported anywhere in the basicam folder.

Run from the project root (the folder containing `basicam`):
    python check_imports.py
"""
import ast
import sys
from importlib import metadata
from pathlib import Path

root = Path(__file__).resolve().parent
pkg_dir = root / "basicam"
local = {p.stem for p in pkg_dir.glob("*.py")} | {"basicam"}
stdlib = set(sys.stdlib_module_names)

found: dict[str, set[str]] = {}
for path in sorted(pkg_dir.rglob("*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top not in stdlib and top not in local:
                found.setdefault(top, set()).add(path.name)

dist_map = metadata.packages_distributions()
print(f"{'import name':<16} {'pip package':<22} {'installed':<12} used in")
print("-" * 80)
for top in sorted(found):
    dists = dist_map.get(top, [])
    dist = dists[0] if dists else "(not installed)"
    ver = metadata.version(dist) if dists else "-"
    print(f"{top:<16} {dist:<22} {ver:<12} {', '.join(sorted(found[top]))}")
