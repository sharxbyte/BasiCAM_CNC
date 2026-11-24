# BasiCAM CNC feature outline

Current state of the program after roadmap v1 (13 blocks), roadmap v2 (23 blocks) and roadmap v3
(11 pre-release fixes), and the safety boundaries it keeps.

## Safety boundaries (never weakened)

- Hard limits and keep-out zones cannot be overridden by any setting, code path, console, editor or manual control. This includes the lathe's moving keep-outs (chuck, jaws, collet, tailstock, steady rest) and rotary swings into keep-outs.
- Soft limits are enforced everywhere; only a per-axis Released switch, with the `release_soft_limits` gate on, bypasses them.
- No tool assignment and no G-code from under-defined or broken (non-construction) sketch geometry, or from a sketch that fails Check Sketch (gaps, overlaps, zero-length or self-crossing entities).
- Every "allow breaking this rule" switch (Settings > Advanced) starts off and needs its warning acknowledged; Beginner mode forces them all off. This includes `laser_test_fire`.
- A running program cannot be edited until the machine and spindle are stopped; it is re-checked before resuming. The tool table and manual auxiliaries cannot be changed while it runs.
- Console and editor input that would collide, cross a hard limit or enter a keep-out zone is refused.
- Laser, plasma and waterjet: the beam / arc / jet is never on during a rapid or a gated traverse, and only over the stock.
- Lathes: the chuck only opens with the spindle stopped.
- One readiness gate decides every send to the controller (buttons, menu, pad, keyboard, console, preflight, programs); refusals always give the reason.
- BasiCAM never turns the controller's hard or soft limits off: the console refuses `$21=0` and `$20=0`; `$5` (limit-pin inversion) changes only through the guided limit-switch test, which then proves every switch again.
- At a limit, a move may stay there or move away; it may never move further into the limit (hard and soft).
- KILL is on the main window (status bar), the jog pad and every window that moves the machine, and is never hidden.

## Modules

| Area | Module | What it does |
|---|---|---|
| Safety switches | `safety.py` | Gate registry, Settings > Advanced, first-run dialog, Beginner mode |
| Terminal | `terminal.py` | Shared log (rotating file), command validation, paste and history rules |
| Operation stack | `operation_stack.py` | Drag-to-reorder widgets and renumbering |
| Colours | `theme.py`, `ui_tips.py` | Colour tokens (red only for errors and keep-out geometry), colour-blind palette, overrides; tool-tip table |
| Display | `display_modes.py` | Five display modes, dashed hidden edges, lighting parameters |
| Import / export | `importers.py`, `exporters.py` | STEP, IGES, STL (planar faces grouped), DXF, DWG (ODA), SLDPRT; placement at the origin / beside parts; STL/STEP export |
| Selection | `face_selection.py` | Box selection by whole faces, vectorised projection |
| Materials and stock | `materials.py`, `stock.py`, `part_materials_dialog.py` | Material library, block / sheet / plate / bar / tube stock, per-part material and thickness |
| Sketch solving | `sketch_solver.py`, `sketch_dialogs.py` | python-solvespace wrapper, relations (incl. collinear, symmetric, on-edge, on-curve, equal radius), dimensions, drag to explore, last good state |
| Sketch drawing | `sketch_interaction.py` | Rubber-band previews, cursor readouts, chaining, relation inference, typed values |
| Smart Dimension | `sketch_dimensions.py` | Dimension type inference, safe arithmetic parser, label geometry |
| Sketch tools | `sketch_tools.py` | Trim, extend, offset, mirror, fillet, chamfer, convert entities, move / copy, fully define |
| Contours | `contours.py` | Regions with islands, open chains, drill points, Check Sketch, pocket rings, compensated profiles |
| Toolpath sketches | `toolpath_sketch.py`, `surfacing.py` | Assign tool on a surface: envelope outside the surface (flat, ball, bull, V-bit), rounding away from the material, shortest-rapid ordering, rough + finish |
| Machines | `machine_profiles.py`, `machine_types.py` | Inventory (unsaved-edit tracking, port and baud), processes, axis layouts, controllers, limits, keep-outs, preflight window, bed fit |
| Windows | `window_fit.py` | Fit every window to its screen, scrolling long forms, remembered sizes, maximised main window |
| Machine-type UI | `ui_visibility.py` | Which commands, panels and operation types show for the loaded machine |
| Connection | `grbl_handshake.py`, `ports_dialog.py`, `limit_test.py` | Boot-banner handshake with retries and diagnostics, firmware detection (GRBL / grblHAL / FluidNC), Auto baud, Test connection, ports window, limit diagnostics and the guided limit-switch test |
| Readiness | `machine_readiness.py` | The single gate and the Ready checklist (connect, identify, alarm, verify, preflight, ready) |
| Manual controller | `jog_pad.py` | Jog pad: X/Y cross with diagonals, Z column, per-axis steps with Custom, Link X/Y, feeds, hold to move, rotary rows, KILL |
| Kinematics | `kinematics.py` | Forward / inverse kinematics for 3-axis, 4th axis, table-table, head-table, lathe, live tooling, U/V taper; 3+2 frames |
| Motion guard | `motion_guard.py` | Tool-envelope checks for moves and programs (HARD / KEEPOUT / COLLISION / SOFT), multi-axis joint moves, tool changes |
| Tool changers | `tool_changer.py` | Changer types, pockets, tool table, change blocks (T/M6 + G43 or M0) |
| Auxiliaries and usage | `auxiliaries.py`, `usage_tracking.py` | Air, gas, coolant, fluid, water, vacuum, dust, fumes; consumable counters and maintenance thresholds |
| 2D cutting | `process_params.py`, `cut_features.py` | Cut charts, kerf compensation, safe leads, tabs / micro-joints, pierces, flash cutting, cut order, taper, beam checks |
| Rotary | `rotary_devices.py`, `wrapping.py`, `job_type.py` | Rotary devices, wrapped and indexed work, flat / round / on a 3D object |
| Nesting | `nesting.py` | True-shape bottom-left nesting with part-in-hole, rotation toggle, grain lock, groups |
| Lathe | `turning_setup.py`, `machine_bodies.py`, `turning.py`, `lathe_tools.py`, `bar_feed.py` | Turning setup (off-centre), moving keep-outs, turning operations and lathe post, bar feeders |
| Positioning | `transforms.py` | Translate, rotate, align, work frame (new geometry, exact undo) |
| Tools | `tooling.py`, `tooling_ui.py` | Tool model measured from the tip (cutting / flute length, neck and reach, corner radius, taper, helix), generated profiles, preview with flutes and dimensions, inventory saved on every change |
| Tool assignment | `assignments.py` | Assignment rules and memory, sketch status, generation gate |
| Cycle time | `cycle_time.py` | Live estimate, multiplier rules, GRBL overrides, F-word scaling, tool-change time, rotary surface distance |
| Keyboard control | `keyboard_control.py` | Keyboard machine-control state machine ([ / ] rotary) |
| Advanced G-code | `gcode_viewer.py`, `gcode_dialogs.py` | Parser, highlighter, editor, resume preamble, section setup sheet |
| Projects | `project_bundle.py` | `.bcam` bundles with `extras.json`, autosave, crash recovery |

## Known limits

- Simultaneous 5-axis contouring is not supported (3+2 positional only). Sinker EDM, text-to-curves and image engraving are not in this version.
- Streaming runs over the GRBL protocol (GRBL, grblHAL, FluidNC); other controllers get G-code files to run on their own software.
- Flute helix geometry is not modelled; tool envelopes come from the tool inventory.
- Surfacing works on a height map from above; undercuts cannot be machined in 3 axes.
- The lathe stock-removal simulation is 2D (half-profile); machine bodies are drawn where they stand, not animated along the program.
- Contour hover-highlighting in the view is not implemented: pick targets in the sketch tree or with Pick sketch feature.
- Visible-only box selection uses the OpenGL depth buffer and is only tested on screen, not headless.
- Travel limits (hard and soft) apply to the X / Y axis position, i.e. the end stops. The tool outline (cutter, shank and holder plus 5 mm) is a separate body that moves with the tool: no part of it may enter a keep-out zone (machine parts) or a fixture (jigs, clamps), and only its cutting length may enter stock, on feed moves. Z min stays tool-aware (stick-out and the clearance below the tip).
- Not yet run end to end on a real machine; see `docs/hardware_test.md` (sections 11-15 cover the v3 connection, limit test, jog pad and preflight).
