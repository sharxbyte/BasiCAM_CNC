# BasiCAM CNC

Desktop CAD/CAM with GRBL machine control, for Windows and Linux.

Source: <https://github.com/sharxbyte/BasiCAM_CNC>

## Development status

- Version: all 13 roadmap blocks complete (2026-11-21).
- Last change (block 13): DWG import through the ODA File Converter, **File > Export part...** (STL / STEP), timestamped GRBL alarm and error reporting, refreshed documentation, final UI audit.
- Open items: none from the roadmap. Quarantined: none.
- **Not yet run against a real machine.** Use `docs/hardware_test.md` for the first session and verify all generated G-code independently.

## What it does

- **Import**: STEP and IGES (exact B-rep), STL (converted to solids when watertight), DXF and DWG linework, SLDPRT on Windows with SolidWorks installed. Unsupported drawing entities are listed after import. **File > Export part...** writes STL or STEP and lists anything that could not be exported.
- **View**: Shaded, Unshaded solid, Transparent shaded, Hidden lines (hidden edges dashed) and Wireframe; only real face boundaries are drawn. **Lighting...** sets brightness, contrast and light direction. Colour-blind palette and colour overrides in Preferences; red is reserved for errors.
- **Sketch**: point, line, rectangle (corner / center / 3 point), circle (center / 3 point), arc (3 point / center) and spline, with snaps and construction geometry. Dimensions take mm, cm, m, in, ft and fractions. Each sketch is solved after every edit (python-solvespace): fully defined, under-defined (yellow, `*`) or could not solve (red). Relations: coincident, horizontal, vertical, parallel, perpendicular, tangent, equal, concentric, midpoint, fix, anchor to origin or model.
- **Position**: translate, rotate, align (face to face, vertex to vertex, face to plane, point to coordinates or a sketch point), work origin and axes from a picked point and edge; arrow keys nudge the selected part when the view is square to a face.
- **Machines**: a machine inventory (travel, feeds, spindle range, homing, keep-out zones), soft limits recorded at the endstops, a preflight check after every connect or alarm, and bed-fit warnings.
- **Operations**: tools only on fully defined sketch entities; depth and entry options (first passes, final pass, ramp, lead-in/out); 3-axis surface roughing and finishing (**Trace curved face...**); setup work offset and heights; a generation gate for untooled sketches; drag to reorder, duplicate, assign to several targets.
- **Motion guard**: every jog, keyboard move, console line, generated program and stream is checked against a tool envelope (radius = diameter/2 + 5 mm). Hard limits and keep-out zones can never be overridden.
- **Run**: live cycle-time estimate, speed multiplier (GRBL overrides up to 200%, more only with a safety switch), keyboard machine control with a ghost preview (virtual by default), and the **Advanced G-code** editor (preview from a line, delete prior operations, resume with a safe preamble, export a section with its setup sheet).
- **Projects**: `.bcam` bundles with undo/redo (50 steps by default), autosave and crash recovery; one shared terminal log for the app and the controller.

## Supported machines and controllers

GRBL 1.1-compatible controllers over USB serial (3-axis mills and routers). Other controllers and protocols are not supported. Feed and spindle overrides follow GRBL 1.1 (10-200%; rapids 25/50/100%).

## Requirements

- Python 3.11 or newer.
- `requirements.txt`: PySide6, pyvista, pyvistaqt, vtk, numpy, trimesh, ezdxf, cadquery-ocp, python-solvespace, shapely, pyserial, pyinstaller, pyvista-cad; pywin32 on Windows.
- `python-solvespace` is GPLv3. Distributing a build that bundles it brings GPL obligations.
- DWG import needs the free ODA File Converter; set its path in **Options > Preferences > DWG converter**.

## Run from source

Windows (PowerShell):

```powershell
cd "BasiCAM CNC"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m basicam
```

Linux:

```bash
cd "BasiCAM CNC"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m basicam
```

## Build a Windows executable

From PowerShell in the project folder, with the virtual environment active:

```powershell
pyinstaller --noconfirm --clean BasiCAM_CNC.spec
```

The one-file, windowed build is written to `dist\BasiCAM CNC.exe`. For a console build that shows start-up errors, set `$env:BASICAM_DIAGNOSTIC = "1"` before running PyInstaller (produces `dist\BasiCAM CNC Diagnostic.exe`). Builds only run on the OS they were built on. Some antivirus products flag unsigned one-file executables; see `ThisIsNotMalware.txt`.

## Safety switches and Beginner mode

At first start-up BasiCAM asks whether you are a new user. **Yes** turns on Beginner mode: every warning and confirmation is on, all safety switches are off, and Settings > Advanced and Advanced G-code are unavailable. Machine control stays available. Turn it off in **Options > Preferences**.

**Options > Settings > Advanced...** holds the switches that allow a rule to be broken: speed multiplier above 200%, G-code for a part larger than the machine in all three dimensions, releasing soft limits or margins under 5 mm, and Advanced G-code mode. Each starts off and needs its warning acknowledged. Hard limits, keep-out zones, collision alarms and code from under-defined sketches can never be overridden.

## First workflow

1. **Open part** (or Import part), then create or import stock under the stock setup.
2. Choose your machine in the Machine dock (add it under **Tool inventory > Machines** first).
3. In **Sketch**, start a sketch on a plane or face, draw the profiles, then dimension and anchor them until nothing is yellow.
4. In **Operations**, pick a tool, set depth and entry, select the sketch as the target and **Add operation to stack**. Use **Trace curved face...** for 3-axis surfaces.
5. Set the work offset and heights in **Setup & Export**, then open the **G-code checklist**.
6. **Generate G-code...** (or the operation row's **G-code** button for one operation). Mark sketches you will tool later as Future operation when asked.
7. To run: connect, complete the preflight check, set soft limits, home or verify position, then stream from the Advanced G-code editor (gate required) or your own sender.

## Machine control

Connect in the Machine dock, wait for GRBL identification, complete the preflight check (tick each axis after moving it through its travel), then home or confirm the position. The header shows Disconnected / Connected - not homed / Connected - ready / RUNNING / ALARM n; alarms and errors are logged with timestamps and a plain-language description. A soft reset is not an emergency stop: use the machine's physical E-stop.

## Tests

`python -m unittest discover tests` runs the suite; Qt tests use `QT_QPA_PLATFORM=offscreen`. `python tools/ui_audit.py --fire` checks UI wiring (not bundled). `docs/hardware_test.md` is the manual hardware checklist.

## Safety note

Generated G-code, previews, simulation, collision checks and feeds/speeds values are aids, not guarantees. Verify toolpaths, tooling, fixturing and machine setup independently before running anything on a machine.
