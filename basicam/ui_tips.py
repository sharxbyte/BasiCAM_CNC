"""Central tool-tip table. Applied once after the main window is built.

Only controls whose tool tip is empty, or equal to their label, are filled in,
so tips written next to the widget code always win.
"""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractButton, QWidget

# Window attribute name -> tool tip.
TIPS_BY_ATTR: dict[str, str] = {
    "note_target": "Part, feature or sketch whose note is shown in the editor below.",
    "operation_kind": "Type of machining operation to add.",
    "pick_feature_button": "Click a sketch feature in the viewport to make it the operation target.",
    "tool_name": "Inventory tool used by the new operation; also updates the tool preview.",
    "tool_diameter": "Cutting diameter used for stepover and feeds.",
    "spindle": "Spindle speed for the new operation.",
    "manual_spindle_box": "Keep the typed spindle speed instead of the recommended one.",
    "warn_2x_box": "Warn before running or exporting with a speed multiplier above 2x.",
    "feed": "Cutting feed rate for the new operation.",
    "stepdown": "Depth removed by each pass.",
    "depth": "Final depth of the operation below the top of stock.",
    "direction": "Climb or conventional milling direction.",
    "setup_orientation_combo": "Which side of the part faces the spindle in the active setup.",
    "post_combo": "Post-processor dialect used for G-code export.",
    "sketch_assign_combo": "Sketch to receive a toolpath.",
    "op_assign_combo": "Operation whose toolpath is assigned to the chosen sketch.",
    "fixture_name_combo": "Fixture type to add to the collision model.",
    "tool_library_combo": "Library cutter to load into the feeds and speeds calculator.",
    "flutes_spin": "Number of cutting flutes used for chip-load calculation.",
    "work_plane": "Plane new sketches are drawn on.",
    "sketch_variant": "Drawing method for the selected sketch tool.",
    "finish_spline_btn": "Close the spline being drawn and add it to the sketch.",
    "dimension_input": "Type a length with units (mm, cm, m, in, ft or fractions) and press Apply.",
    "sketch_visible": "Show or hide all sketch entities in the viewport.",
    "construction_visible": "Show or hide construction geometry in the viewport.",
    "sketch_tree": "Sketches and dimensions; right-click for tool, operation and note actions.",
    "display_mode": "How parts are drawn in the viewport.",
    "measure_button": "Click two points in the viewport to measure the distance.",
    "stock_material": "Stock material; sets feeds, speeds and default overlap.",
    "port_combo": "Serial port of the GRBL controller.",
    "baud_combo": "Serial speed; GRBL 1.1 normally uses 115200.",
    "connect_button": "Open or close the serial connection to the controller.",
    "position_verified_box": "Confirm the machine is homed or its position checked; motion stays off until then.",
    "jog_distance": "Distance of one jog step.",
    "jog_feed": "Feed rate for jog moves.",
    "home_button": "Run the GRBL homing cycle ($H); the machine moves toward its switches.",
    "unlock_button": "Clear a GRBL alarm lock ($X) after the cause has been fixed.",
    "hold_button": "Pause motion with a feed hold (!).",
    "resume_button": "Resume after a feed hold (~).",
    "kill_button": "Stop the spindle and motion, then retract; use the physical E-stop in an emergency.",
    "reset_controller_button": "Soft-reset the controller (Ctrl-X); position may be lost. Not an E-stop.",
    "spindle_rpm": "Spindle speed for Spindle on.",
    "spindle_start_button": "Start the spindle clockwise at the set speed.",
    "spindle_stop_button": "Stop the spindle (M5).",
    "manual_cut_dir": "Axis and direction of the manual cut.",
    "manual_cut_distance": "Length of the manual cut move.",
    "manual_cut_feed": "Feed rate of the manual cut.",
    "manual_cut_rpm": "Spindle speed for the manual cut.",
    "manual_cut_spindle_running_box": "Skip spindle start because it is already running.",
    "export_button": "Generate G-code for every operation and save it to a file.",
    "through_toggle": "When enabled, box selection includes geometry behind the visible surface.",
}

# Visible label text -> tool tip (for controls not stored on the window).
TIPS_BY_TEXT: dict[str, str] = {
    "Show / refresh 3D toolpath preview": "Draw the toolpaths of all operations in the viewport.",
    "Clear 3D toolpath preview": "Remove toolpath lines from the viewport.",
    "Play / pause simulation": "Start or pause the toolpath simulation.",
    "Preview tool in scene": "Show the current tool at the simulation position.",
    "G-code checklist…": "List the setup data still missing before G-code can be generated.",
    "+ New setup": "Add a machining setup with its own orientation and operations.",
    "Estimate Stock Removal  Features": "Estimate removed volume and recognised features from part and stock.",
    "Export CAM Package (.nc + Sheets)": "Write the G-code, setup sheet and tool list to a folder.",
    "Set Post-Processor": "Use the selected post-processor for export.",
    "Reset": "Return the simulation to the first segment.",
    "Play / Pause Sim": "Start or pause the toolpath simulation.",
    "Show / Refresh 3D Toolpath Preview": "Draw the toolpaths of all operations in the viewport.",
    "Clear 3D Toolpath Preview": "Remove toolpath lines from the viewport.",
    "Assign Toolpath to Sketch": "Link the chosen operation's toolpath to the chosen sketch.",
    "Add Fixture": "Add a fixture box of this size to the collision model.",
    "Check Collisions  Envelope": "Check toolpaths against fixtures and the machine travel.",
    "Load Selected Tool": "Copy the library tool's data into the calculator.",
    "Calculate Feeds  Speeds": "Calculate spindle speed and feed for the tool and material.",
    "Preview Tool Cylinder in Scene": "Show the current tool at the simulation position.",
    "Point": "Place sketch points (P).",
    "Line": "Draw sketch lines (L).",
    "Rectangle": "Draw a rectangle (R).",
    "Circle": "Draw a circle (C).",
    "Curve": "Draw an arc (A).",
    "Spline": "Draw a spline through clicked points.",
    "Grid": "Snap to the grid.",
    "Endpoint": "Snap to entity endpoints.",
    "Midpoint": "Snap to entity midpoints.",
    "Center": "Snap to circle and arc centres.",
    "Intersection": "Snap to intersections between entities.",
    "On surface": "Snap onto the picked model surface.",
    "Apply": "Apply the typed dimension to the selected entity.",
    "Open part": "Open a part in a new, empty workspace.",
    "Tool inventory": "Edit the tool and machine inventory.",
    "Donate": "Open the project's donation page in your browser.",
    "＋  Import part": "Add a part to the current workspace.",
    "Create box": "Create box stock from the X/Y/Z sizes.",
    "Import billet": "Import a shape to use as stock.",
    "Zero work (G10)": "Set the current position as work zero (G10 L20).",
    "Run cut": "Run the manual cut move after confirmation.",
}

# Action text -> tool tip.
ACTION_TIPS: dict[str, str] = {
    "Save Project (.bcam)": "Save the workspace, geometry and settings to a .bcam file.",
    "Save Project As (.bcam)…": "Save the workspace to a new .bcam file.",
    "Open Project (.bcam)…": "Open a saved .bcam workspace.",
    "CAM Workflows": "Show or hide the CAM workflow tabs.",
    "Machine & Console": "Show or hide the machine controls and console.",
    "Terminal": "Show or hide the bottom terminal.",
    "GitHub Repository": "Open the project's source page in your browser.",
    "Delete": "Delete the selection, layer by layer.",
    "Cancel / back out": "Cancel the current tool or selection.",
    "New sketch / finish sketch": "Start a sketch on the work plane, or finish the active one.",
    "Line": "Draw sketch lines.",
    "Circle (center point)": "Draw a circle from its centre.",
    "Rectangle (corner)": "Draw a rectangle from two corners.",
    "Point": "Place sketch points.",
    "Arc (3 point)": "Draw an arc through three points.",
    "Normal to selected": "Look straight at the selected face or curve.",
    "Color-blind friendly axes": "Use a colour-blind safe palette for the axes.",
    "Toggle tool tips": "Show or hide tool tips everywhere.",
}


def _label(obj) -> str:
    return obj.text().replace("&", "").strip() if hasattr(obj, "text") else ""


def _needs_tip(obj) -> bool:
    tip = obj.toolTip().strip()
    return not tip or tip.casefold() == _label(obj).casefold()


def apply_tooltips(window) -> int:
    """Fill missing tool tips; returns how many were set."""
    count = 0
    for name, tip in TIPS_BY_ATTR.items():
        obj = getattr(window, name, None)
        if isinstance(obj, (QWidget, QAction)) and _needs_tip(obj):
            obj.setToolTip(tip)
            count += 1
    for widget in window.findChildren(QAbstractButton):
        tip = TIPS_BY_TEXT.get(_label(widget))
        if tip and _needs_tip(widget):
            widget.setToolTip(tip)
            count += 1
    for action in window.findChildren(QAction):
        tip = ACTION_TIPS.get(_label(action))
        if tip and _needs_tip(action):
            action.setToolTip(tip)
            count += 1
    return count
