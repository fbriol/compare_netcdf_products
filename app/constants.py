"""Constants used across the application."""

from __future__ import annotations

#: Row status colors for the comparison table.
STATUS_COLORS = {
    "equal": "#ccffcc",
    "almost_equal": "#006400",
    "different": "#ffa500",
    "masked": "#d3d3d3",
    "missing_in_target": "#e57373",
    "missing_in_source": "#7986cb",
}

#: Text colors for the comparison table (white on dark backgrounds).
STATUS_TEXT_COLORS = {
    "equal": "#000000",
    "almost_equal": "#ffffff",
    "different": "#000000",
    "masked": "#000000",
    "missing_in_target": "#ffffff",
    "missing_in_source": "#ffffff",
}

#: Maximum number of points to display in scatter plots before downsampling.
MAX_DISPLAY_POINTS = 20_000

#: Maximum number of variables to keep in the LRU cache.
MAX_VARIABLE_CACHE_SIZE = 5

#: Legend data for the comparison table.
LEGEND_DATA = [
    ("Values are strictly equal", "equal"),
    ("Values are almost equal (within tolerance)", "almost_equal"),
    ("Values are different", "different"),
    ("Values are all masked (no data)", "masked"),
    ("Variable missing in target file", "missing_in_target"),
    ("Variable missing in source file", "missing_in_source"),
]
