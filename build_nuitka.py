"""Build the Windows program with Nuitka as one folder (Nuitka's standalone mode, the --onedir layout).

    python build_nuitka.py              # dist\\BasiCAM CNC\\BasiCAM CNC.exe plus its DLLs
    python build_nuitka.py --diagnostic # same, with a console window for tracebacks
    python build_nuitka.py --zip        # also dist\\BasiCAM-CNC-<version>-windows-x64.zip

The same script runs locally and in .github/workflows/build.yml, so the folder SignPath signs is
the folder this script produces. Version and product details come from basicam/about.py and end up
in the Windows file properties of the exe.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
FOLDER = "BasiCAM CNC"
EXE = "BasiCAM CNC.exe"

# Shipped next to the exe; LICENSE and TERMS_OF_USE.md are also read by Help > Terms of Use and License.
DATA_FILES = ("LICENSE", "TERMS_OF_USE.md", "THIRD_PARTY_NOTICES.md", "BCCIcon.ico", "ThisIsNotMalware.txt")


def app_version() -> str:
    text = (ROOT / "basicam" / "about.py").read_text(encoding="utf-8")
    return re.search(r'APP_VERSION = "([\d.]+)"', text).group(1)


def windows_icon() -> Path:
    """BCCIcon.ico is PNG data under an .ico name (Qt reads it fine at run time); Windows resources
    need a real ICO, so a multi-size one is written next to the build output."""
    source = ROOT / "BCCIcon.ico"
    if source.read_bytes()[:4] == b"\x00\x00\x01\x00":
        return source
    from PIL import Image

    target = DIST / "build-icon" / "BasiCAM CNC.ico"
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGBA")
    side = max(image.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    square = square.resize((256, 256), Image.Resampling.LANCZOS)
    square.save(target, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return target


def nuitka_command(version: str, diagnostic: bool) -> list[str]:
    numeric = ".".join((version.split(".") + ["0", "0", "0"])[:4])
    cmd = [
        sys.executable, "-m", "nuitka",
        "--mode=standalone",  # one folder, not a self-extracting single file
        "--assume-yes-for-downloads",
        "--msvc=latest",
        f"--output-dir={DIST}",
        f"--output-folder-name={FOLDER}",
        f"--output-filename={EXE}",
        "--enable-plugin=pyside6",
        "--include-package=basicam",
        "--include-package=vtkmodules",  # pyvista imports VTK modules by name at run time
        "--include-package=pyvistaqt",
        "--include-package=python_solvespace",
        "--include-package=OCP",
        "--include-module=serial.tools.list_ports",
        "--include-package-data=pyvista",
        "--include-package-data=trimesh",
        "--include-package-data=ezdxf",
        "--include-package-data=shapely",
        "--nofollow-import-to=tests",
        "--nofollow-import-to=tools",
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=PyInstaller",
        "--nofollow-import-to=nuitka",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=jupyter_client",
        "--nofollow-import-to=trame",
        f"--windows-icon-from-ico={windows_icon()}",
        "--windows-console-mode=" + ("force" if diagnostic else "disable"),
        "--company-name=BasiCAM CNC",
        "--product-name=BasiCAM CNC",
        f"--file-version={numeric}",
        f"--product-version={numeric}",
        "--file-description=BasiCAM CNC - CAD/CAM and GRBL machine control",
        "--copyright=Copyright (C) 2026 BasiCAM CNC contributors. GPL-3.0-or-later.",
        f"--report={DIST / 'nuitka-report.xml'}",
    ]
    if sys.platform == "win32":
        cmd += ["--include-module=win32com.client", "--include-module=pythoncom", "--include-module=pywintypes"]
    for name in DATA_FILES:
        cmd.append(f"--include-data-files={ROOT / name}={name}")
    cmd.append(str(ROOT / "basicam" / "__main__.py"))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic", action="store_true", help="console window for tracebacks")
    parser.add_argument("--zip", action="store_true", help="also zip the folder for release")
    args = parser.parse_args()
    version = app_version()
    for previous in (DIST / FOLDER, DIST / f"{FOLDER}.dist"):
        if previous.exists():
            shutil.rmtree(previous)  # only the previous build output of this script
    result = subprocess.call(nuitka_command(version, args.diagnostic), cwd=ROOT)
    if result != 0:
        print(f"Nuitka failed with exit code {result}")
        return result
    return finish(version, args.zip)


def finish(version: str, make_zip: bool) -> int:
    """Nuitka always names the folder '<name>.dist'; it is renamed to the plain name, then zipped."""
    target = DIST / FOLDER
    built = DIST / f"{FOLDER}.dist"
    if built.exists():
        if target.exists():
            shutil.rmtree(target)
        built.rename(target)
    exe = target / EXE
    if not exe.exists():
        print(f"Build finished but {exe} is missing")
        return 1
    print(f"Built: {exe}")
    if make_zip:
        archive = DIST / f"BasiCAM-CNC-{version}-windows-x64"
        shutil.make_archive(str(archive), "zip", root_dir=DIST, base_dir=FOLDER)
        print(f"Zipped: {archive}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
