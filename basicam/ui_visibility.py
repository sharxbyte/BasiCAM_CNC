"""Which commands, panels and fields apply to the loaded machine (roadmap v3 Block 5).

Options that only some machines can use are hidden, not just disabled, when the loaded machine
cannot use them. A rule is a trait name, "!trait" (the machine must not have it), or a tuple of
rules of which any may match. With no machine loaded the mill / router set shows.
"""

from __future__ import annotations

from basicam.machine_types import LATHE_PROCESSES, TWO_D_PROCESSES

ROTARY_LAYOUTS = ("gantry3_rotary", "table_table5", "head_table5", "lathe_live")
FIVE_AXIS_LAYOUTS = ("table_table5", "head_table5")
LATHE_LAYOUTS = ("lathe2", "lathe_live")
LASER_PROCESSES = ("laser", "rotary_laser", "laser_lathe")

# Commands registered with _add_command (by name).
COMMAND_RULES: dict[str, object] = {
    "laser_test_fire": "laser",
    "rotary_index": "rotary",
    "orient_3p2": "five_axis",
    "nest_parts": ("two_d", "sheet"),
    "turning_setup": "lathe",
    "turning_side_view": "lathe",
    "turning_silhouette": "lathe",
    "chuck_open": "lathe",
    "production_run": "bar_feed",
}
MENU_RULES: dict[str, object] = {"Turning": "lathe"}
# Machine dock tabs (by title).
PANEL_RULES: dict[str, object] = {"Tool table": "changer"}
# Window attributes (widgets) shown only when the rule matches.
WIDGET_RULES: dict[str, object] = {
    "wrap_rotary": "rotary",
    "spindle_start_button": "spindle",
}
# Never hidden, whatever the machine (KILL, feed hold, spindle / beam stop).
NEVER_HIDDEN = ("kill_button", "hold_button", "spindle_stop_button")

MILL_KINDS = ("Rough", "Finish", "Pocket", "Profile", "Trace", "Drill", "Ream", "Tap", "Engrave")
CUT_KINDS = ("Cut", "Engrave", "Trace")
LATHE_KINDS = ("Face", "Rough turn", "Finish turn", "Groove", "Part off", "Centre drill", "Thread", "Drill")
ALL_KINDS = MILL_KINDS + tuple(k for k in CUT_KINDS + LATHE_KINDS if k not in MILL_KINDS)

TRAIT_NAMES = {
    "mill": "mill or router",
    "two_d": "laser, plasma, waterjet or EDM",
    "laser": "laser",
    "lathe": "lathe",
    "rotary": "rotary axis",
    "five_axis": "5-axis",
    "changer": "tool changer",
    "bar_feed": "bar feeder",
    "sheet": "sheet or plate stock",
    "spindle": "spindle",
}


def machine_traits(machine, stock_kind: str = "") -> set[str]:
    """What the loaded machine (and stock) can do."""
    traits: set[str] = set()
    if str(stock_kind) in ("sheet", "plate"):
        traits.add("sheet")
    if machine is None:
        return traits | {"mill", "spindle"}
    process = str(getattr(machine, "process", "mill") or "mill")
    layout = str(getattr(machine, "layout", "gantry3") or "gantry3")
    if process in ("mill", "mill_turn"):
        traits.add("mill")
    if process in TWO_D_PROCESSES or process == "rotary_laser":
        traits.add("two_d")
    if process in LASER_PROCESSES:
        traits.add("laser")
    if process in LATHE_PROCESSES or layout in LATHE_LAYOUTS:
        traits.add("lathe")
    if layout in ROTARY_LAYOUTS or process == "rotary_laser" or getattr(machine, "rotary_devices", None):
        traits.add("rotary")
    if layout in FIVE_AXIS_LAYOUTS:
        traits.add("five_axis")
    changer = getattr(machine, "tool_changer", None)
    if changer is not None and str(getattr(changer, "kind", "none")) not in ("none", "manual", ""):
        traits.add("changer")
    feed = getattr(machine, "bar_feed", None)
    if feed is not None and str(getattr(feed, "kind", "none")) not in ("none", ""):
        traits.add("bar_feed")
    if "two_d" not in traits or "lathe" in traits:
        traits.add("spindle")
    return traits


def visible(rule, traits: set[str]) -> bool:
    if rule is None:
        return True
    if isinstance(rule, tuple):
        return any(visible(r, traits) for r in rule)
    rule = str(rule)
    if rule.startswith("!"):
        return rule[1:] not in traits
    return rule in traits


def operation_kinds(traits: set[str]) -> list[str]:
    """Operation types the machine can run, in menu order."""
    kinds: list[str] = []
    if "mill" in traits:
        kinds += MILL_KINDS
    if "two_d" in traits:
        kinds += CUT_KINDS
    if "lathe" in traits:
        kinds += LATHE_KINDS
    if not kinds:
        kinds = list(MILL_KINDS)
    ordered = [k for k in ALL_KINDS if k in kinds]
    return ordered


def needs_text(kind: str) -> str:
    """'needs a lathe machine' for an operation the loaded machine cannot run."""
    if kind in LATHE_KINDS and kind not in MILL_KINDS:
        return "needs a lathe machine"
    if kind == "Cut":
        return "needs a laser, plasma, waterjet or EDM machine"
    return "needs a mill or router"
