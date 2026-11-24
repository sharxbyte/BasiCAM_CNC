# BasiCAM CNC

Desktop CAD/CAM with GRBL machine control, for Windows and Linux.

Source: <https://github.com/sharxbyte/BasiCAM_CNC>

## Development status

- Version: 0.24.0 pre-release (2026-11-24). Roadmaps v1 (13 blocks), v2 (23 blocks) and v3 (11 pre-release fixes) complete.
- Last change: X / Y travel limits apply to the axis position; the tool outline moves with the tool and may never touch keep-outs or fixtures (jigs, clamps), and only its cutting length may cut stock. Fixed the start-up crash when restoring the last session after an unclean exit. An unnamed project takes the name of the part opened. 227 tests.
- Open items: none from the roadmap. Quarantined: none.
- **Not yet run against a real machine.** Use `docs/hardware_test.md` for the first session and verify all generated G-code independently.

## What it does

- **Import**: STEP and IGES (exact B-rep), STL (converted to solids when watertight; flat areas grouped into faces), DXF and DWG linework, SLDPRT on Windows with SolidWorks installed. The first part goes to 0,0,0 at its bounding-box corner (**Keep file coordinates** keeps the file's position); later parts are placed beside the existing ones. Several parts can be opened at once, with one material and thickness for all or a list per part. Box selection takes whole faces. **File > Export part...** writes STL or STEP.
- **Stock and materials**: block, sheet, plate, bar and tube stock (**Stock...**); a material library (Tool inventory > Materials). **Tools > Nest parts on the stock...** nests parts true-shape on sheets (part-in-hole, spacing 2 x kerf by default, rotation off unless allowed, grain lock, separate stocks per material).
- **Windows**: the main window opens maximised; every window fits the screen it opens on, can be resized (long forms scroll, Save / OK stay visible) and remembers its size.
- **View**: Shaded, Unshaded solid, Transparent shaded, Hidden lines and Wireframe; **Lighting...**; colour-blind palette. Red is reserved for errors and keep-out geometry (jaws, chuck faces).
- **Sketch** (see `docs/sketching.md`): SolidWorks-style drawing with rubber-band previews, line chaining, typed lengths and screen-distance snapping; snapping to a model vertex defines the point there. Point, line, rectangle, circle, arc, spline, slot, polygon, centerline. Relations panel with glyphs, drag to explore, **Smart Dimension (D)** with arithmetic, and Trim, Extend, Offset, Mirror, Fillet, Chamfer, Convert Entities, Move/Copy and Fully Define.
- **Operations**: tools only on fully defined sketches that pass Check Sketch. Pocket (around islands), Profile (inside / outside / on), Trace, Drill (pecks), Engrave, and **Assign tool...** on a surface, which builds a toolpath sketch outside the surface (stepover 60% of the tool, rounded away from the material, rough then 3-axis finish).
- **Tool inventory** (Ctrl+T): tools are defined by catalogue dimensions measured from the tip (cutting diameter, shank, cutting length, flute length, neck and reach, overall length, corner radius, taper, flutes and helix); the preview shows the flutes with cutting edges in orange and can show where each size is measured. One list of tools is used everywhere (operations, Tools & Speeds, tool table); every Add / update and Remove is saved at once; a project that uses a tool this computer does not have offers to add it.
- **Machine inventory** (Tool inventory > Machines): edits are never lost silently (Save / Discard / Cancel), Enter never adds or removes a machine, and each machine keeps its port and baud rate with a **Connect** button.
- **Only what applies**: menus, commands, panels and operation types follow the loaded machine (Turning only for lathes, Laser test fire only for lasers, and so on); **Options > Preferences > Show options for all machine types** shows everything. Operations of another machine type in a project are kept and marked.
- **Machines**: mill / router, laser, plasma, waterjet, wire EDM, lathe, mill-turn, rotary laser and laser lathe; 3-axis, 4th-axis, 5-axis (3+2), lathe and U/V layouts with an axis table and kinematics; tool changers with a tool table; auxiliaries (air, gas, coolant, cutting fluid, water, vacuum, dust, fumes); usage and maintenance counters per machine and per job.
- **Cutting**: cut charts per process, material and thickness; kerf compensation; leads placed for the most clearance from other parts and features (never closer than half the kerf plus a margin) and always over the stock; tabs and micro-joints (2 x kerf by default); pierces off the contour; flash cutting for lasers with fast beam gating; inside-first cut order; taper compensation. The beam is never on during a rapid or outside the stock. **Laser test fire** needs its safety switch.
- **Rotary and lathe**: rotary axis setup (headstock, throw, chuck / collet / rollers, jaws, tailstock, steady rest, speed control); flat / round / on-a-3D-object prompt; wrapped and indexed rotary work (G93 feeds); 3+2 positioning to a face; turning setup with off-centre turning and a side view; facing, roughing, nose-radius-compensated finishing, grooving, parting, drilling and threading (where supported); lathe post (G18, diameter mode, G96 / G95); moving keep-outs for chuck, jaws, tailstock and steady rest; bar feeders with production runs.
- **Position**: translate, rotate, align, work origin and axes; arrow keys nudge the selected part.
- **Motion guard**: every jog, keyboard move (including the rotary axis), console line, generated program and stream is checked: travel limits on the axis position, rotary limits, and the tool (cutter, shank and holder) as a moving body that may never enter a keep-out zone or fixture; only its cutting length may cut stock, and only on feed moves. Hard limits and keep-outs can never be overridden.
- **Run**: live cycle-time estimate (tool changes, rotary moves), speed multiplier, keyboard machine control, the **Advanced G-code** editor.
- **Projects**: `.bcam` bundles with undo/redo, autosave and crash recovery; one shared terminal log.

## Supported machines and controllers

Streamed over USB serial: GRBL 1.1 (3 axes), grblHAL and FluidNC (up to 6 axes; grblHAL lathe mode). Other controllers (LinuxCNC, industrial laser, waterjet and EDM controllers) get G-code files to run on their own software. A machine whose axes do not match the controller is refused.

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


**Machine menu**: Machine inventory, Load last machine, Connect / Disconnect, **Ports...** (lists every serial port with its USB chip, VID:PID and serial number and refreshes when a board is plugged in; Test, Use for this machine, Connect), **Test connection** (identifies the firmware, options, axes, state and limit pins without sending motion), **Reset connection** (closes and reopens the port; the position must be verified again) and **Manual controller**. Jog, home, hold and resume stay disabled until a controller is connected and identified.

**Connecting**: the **Baud rate** is the serial speed set in the controller firmware (GRBL and grblHAL usually 115200); **Auto** tries 115200, 250000, 230400, 57600, 38400, 19200 and 9600 and keeps the one that answers. BasiCAM waits for the controller's start-up message (boards that reset when the port opens need 1-2 s), then identifies GRBL 1.1, grblHAL or FluidNC. If nothing valid arrives it says why (wrong port or baud, port open in another program, not GRBL firmware, charge-only cable) and shows the bytes received. No status report for 5 s shows **Controller not responding** and holds any running program.

**Alarms and limit switches**: alarms are explained for the identified firmware, with what to do next. The machine panel shows which inputs are active (limits, probe, door...). When every limit reads active with nothing pressed, BasiCAM explains the likely causes (inverted pins, normally-closed switches, floating inputs, noise) and the `$5`, `$21`, `$22` values; the **Limit-switch test** has you press each switch and shows it change, and can invert the limit pins (`$5`) when they all read active at rest, then tests every switch again. The console refuses `$21=0` and `$20=0` (limits off) and `$5` outside the test.

**From connected to moving**: the machine panel's **Ready checklist** shows Connected, Firmware identified, Not in alarm, Position verified, Preflight done and Ready, with a button for the next step (Connect, Unlock, Home, Verify position, Run preflight, Limit-switch test). The same check decides every button, menu item, jog, keyboard move, console line and program run, and anything refused says why in the terminal and the status bar. After the position is verified the preflight window opens and its jogs work; normal jogs, the spindle and programs need the preflight to be done. Home works while the controller is in a homing alarm. A machine without homing switches (untick **Homing switches fitted** in the machine inventory) is verified by saying where the tool is (home corner, work zero or typed machine coordinates); the DRO then reads "position set by you" until a reset, alarm or disconnect. Rotary axes the controller reports can be jogged.

**Manual controller** (Machine > Manual controller): X / Y arrows in a cross (the corners move X and Y together) with Z arrows beside them; a step for each axis (0.01 to 100 mm, or 0.001 to 1 in) plus **Custom...**, which accepts units (`2.5`, `0.05in`, `1/8"`); **Link X/Y** keeps the X and Y steps equal; XY and Z feeds (limited to the machine's rates); **Flip X/Y arrows** when the machine moves the other way; **Hold to move** (release to stop); rows for rotary axes; **XY 0** goes to work zero at a safe height. Steps and feeds are kept per machine, and the keyboard arrows use the same steps (Shift x10, Ctrl /10). **KILL** is on the pad, in the status bar of the main window (always visible) and in every window that moves the machine.

**Preflight**: after the position is verified the preflight window opens beside the main window (the console and DRO stay usable). It has the jog pad, Home, Unlock, Feed hold and KILL, and shows each axis's machine position and limit input. Move each axis both ways (pad or keyboard) and its tick is offered; tick every axis to enable normal jogs, the spindle and programs.
## Tests

`python -m unittest discover tests` runs the suite; Qt tests use `QT_QPA_PLATFORM=offscreen`. `python tools/ui_audit.py --fire` checks UI wiring (not bundled). `docs/hardware_test.md` is the manual hardware checklist.

## Safety note

Generated G-code, previews, simulation, collision checks and feeds/speeds values are aids, not guarantees. Verify toolpaths, tooling, fixturing and machine setup independently before running anything on a machine.
