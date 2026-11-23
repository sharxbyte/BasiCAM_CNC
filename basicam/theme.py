"""Central colour table.

Rule: red is reserved for the `error` token. No other default colour (faces,
setup palette, highlights, toolpaths, axes) is in the red hue range.
`color(name)` returns the active colour; `activate(settings)` applies the
colour-blind palette and user overrides.
"""

from __future__ import annotations

import colorsys
import re

TOKENS: dict[str, str] = {
    # UI chrome
    "background": "#13191b",
    "panel": "#1c2426",
    "panel_alt": "#222c2e",
    "line": "#344143",
    "text": "#e4e9e5",
    "muted": "#9ba9a4",
    "accent": "#c2df54",
    # viewport
    "viewport_bg": "#161f21",
    "face": "#8a9296",
    "edge": "#9aa7ab",
    "edge_hidden": "#5d6a6d",
    "highlight": "#4cc9f0",
    "selection": "#c2df54",
    "grid": "#405052",
    "grid_label": "#7a8a8c",
    "origin": "#f2f4f3",
    "axis_x": "#f28c28",
    "axis_y": "#3fb950",
    "axis_z": "#4c8dff",
    "toolpath_cut": "#ff9f1c",
    "toolpath_rapid": "#b07cff",
    "sim_marker": "#06d6a0",
    "tool": "#c2df54",
    "reference": "#8c9a9e",
    # states
    "ok": "#3fb950",
    "warning": "#f2c94c",
    "error": "#ff4d4d",
    "keepout": "#ff4d4d",  # jaws, chuck faces, clamps: no-go geometry (red by request)
    # sketch states
    "sketch_defined": "#e4e9e5",
    "sketch_underdefined": "#f2c94c",
}

COLORBLIND_TOKENS: dict[str, str] = {
    # Okabe-Ito palette
    "axis_x": "#E69F00",
    "axis_y": "#0072B2",
    "axis_z": "#CC79A7",
    "highlight": "#56B4E9",
    "selection": "#F0E442",
    "toolpath_cut": "#E69F00",
    "toolpath_rapid": "#56B4E9",
    "sim_marker": "#009E73",
    "ok": "#009E73",
    "warning": "#F0E442",
    "error": "#D55E00",
    "keepout": "#D55E00",
    "sketch_underdefined": "#F0E442",
    "accent": "#F0E442",
    "tool": "#56B4E9",
}

SETUP_PALETTE_DEFAULT: tuple[str, ...] = ("#c2df54", "#68b5e8", "#e89f54", "#c58be8", "#5ad2b0", "#d6b58a")
SETUP_PALETTE_COLORBLIND: tuple[str, ...] = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#CC79A7")

ERROR_TOKENS = frozenset({"error", "keepout"})  # the only tokens that may be red (errors, keep-out geometry)
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

ACTIVE: dict[str, str] = dict(TOKENS)
ACTIVE_SETUP_PALETTE: list[str] = list(SETUP_PALETTE_DEFAULT)


def is_red(hex_color: str) -> bool:
    """True if the colour is in the red hue range (hue <= 15 or >= 340 deg, saturated, not dark)."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360.0
    return (hue <= 15.0 or hue >= 340.0) and s >= 0.35 and v >= 0.35


def colorblind_variant(tokens: dict[str, str] | None = None) -> dict[str, str]:
    result = dict(tokens or TOKENS)
    result.update(COLORBLIND_TOKENS)
    return result


def load_overrides(settings) -> dict[str, str]:
    """User colour overrides from QSettings `colors/<token>`; red is refused except for `error`."""
    overrides: dict[str, str] = {}
    if settings is None:
        return overrides
    for key in TOKENS:
        raw = settings.value(f"colors/{key}", "")
        text = str(raw or "").strip()
        if not _HEX.match(text):
            continue
        if key not in ERROR_TOKENS and is_red(text):
            continue
        overrides[key] = text
    return overrides


def palette(settings=None, colorblind: bool | None = None) -> dict[str, str]:
    if colorblind is None:
        colorblind = bool(settings is not None and _as_bool(settings.value("view/colorblind_axes", False)))
    base = colorblind_variant() if colorblind else dict(TOKENS)
    base.update(load_overrides(settings))
    return base


def activate(settings=None, colorblind: bool | None = None) -> dict[str, str]:
    ACTIVE.clear()
    ACTIVE.update(palette(settings, colorblind))
    if colorblind is None:
        colorblind = bool(settings is not None and _as_bool(settings.value("view/colorblind_axes", False)))
    ACTIVE_SETUP_PALETTE[:] = list(SETUP_PALETTE_COLORBLIND if colorblind else SETUP_PALETTE_DEFAULT)
    _sync_legacy()
    return ACTIVE


def color(name: str) -> str:
    return ACTIVE.get(name, TOKENS.get(name, "#ffffff"))


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# Legacy dict used by the stylesheet and older code; kept in sync with ACTIVE.
COLORS: dict[str, str] = {}


def _sync_legacy() -> None:
    COLORS.clear()
    COLORS.update(
        {
            "background": ACTIVE["background"],
            "panel": ACTIVE["panel"],
            "panel_alt": ACTIVE["panel_alt"],
            "line": ACTIVE["line"],
            "text": ACTIVE["text"],
            "muted": ACTIVE["muted"],
            "accent": ACTIVE["accent"],
            "warning": ACTIVE["warning"],
            "red": ACTIVE["error"],
            "error": ACTIVE["error"],
        }
    )


_sync_legacy()
