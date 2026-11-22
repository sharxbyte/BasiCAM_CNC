# BasiCAM CNC feature outline

Current state of the program after roadmap blocks 1-13, and the safety boundaries it keeps.

## Safety boundaries (never weakened)

- Hard limits and keep-out zones cannot be overridden by any setting, code path, console, editor or manual control.
- Soft limits are enforced everywhere; only a per-axis Released switch, with the `release_soft_limits` gate on, bypasses them.
- No tool assignment and no G-code from under-defined or broken (non-construction) sketch geometry.
- Every "allow breaking this rule" switch (Settings > Advanced) starts off and needs its warning acknowledged; Beginner mode forces them all off.
- A running program cannot be edited until the machine and spindle are stopped; it is re-checked before resuming.
- Console and editor input that would collide, cross a hard limit or enter a keep-out zone is refused.

## Modules

| Area | Module | What it does |
|---|---|---|
| Safety switches | `safety.py` | Gate registry, Settings > Advanced, first-run dialog, Beginner mode |
| Terminal | `terminal.py` | Shared log (rotating file), command validation, paste and history rules |
| Operation stack | `operation_stack.py` | Drag-to-reorder widgets and renumbering |
| Colours | `theme.py`, `ui_tips.py` | Colour tokens, colour-blind palette, overrides; tool-tip table |
| Display | `display_modes.py` | Five display modes, dashed hidden edges, lighting parameters |
| Machines | `machine_profiles.py` | Machine inventory, soft/hard limits, keep-out zones, preflight, bed fit |
| Motion guard | `motion_guard.py` | Tool-envelope checks for moves and whole programs (HARD / KEEPOUT / COLLISION / SOFT) |
| Sketch solving | `sketch_solver.py`, `sketch_dialogs.py` | python-solvespace wrapper, relations, last good state, cascade delete |
| Positioning | `transforms.py` | Translate, rotate, align, work frame (new geometry, exact undo) |
| Tool assignment | `assignments.py` | Assignment rules and memory, sketch status, generation gate |
| Surfacing | `surfacing.py` | 3-axis roughing/finishing that never cuts into the surface (shapely) |
| Cycle time | `cycle_time.py` | Live estimate, multiplier rules, GRBL overrides, F-word scaling |
| Keyboard control | `keyboard_control.py` | Keyboard machine-control state machine |
| Advanced G-code | `gcode_viewer.py`, `gcode_dialogs.py` | Parser, highlighter, editor, resume preamble, section setup sheet |
| Import / export | `importers.py`, `exporters.py` | STEP, IGES, STL, DXF, DWG (ODA), SLDPRT; STL/STEP export |
| Projects | `project_bundle.py` | `.bcam` bundles with `extras.json`, autosave, crash recovery |

## Known limits

- Only GRBL 1.1-compatible controllers.
- Flute helix geometry is not modelled; tool envelopes come from the tool inventory.
- Surfacing works on a height map from above; undercuts cannot be machined in 3 axes.
- Not yet run end to end on a real machine; see `docs/hardware_test.md`.
