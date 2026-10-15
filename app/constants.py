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

#: Figure labels produced by ``plots.build_all_figures``, in display order.
PLOT_LABELS = (
    "Geographical Position",
    "Signal Comparison",
    "Signal Difference",
    "Scatter Plot",
    "Histogram",
    "Exclusive Undefined Positions",
    "Statistics",
)

#: Plots shown when a variable view opens. The difference is shown first:
#: superimposed source/target curves usually hide differences that are
#: tiny compared with the signal.
DEFAULT_VISIBLE_PLOTS = ("Signal Difference",)

#: Default tolerance, in storage steps (``scale_factor``), under which a
#: difference is classified ``almost_equal``. One step absorbs a rounding
#: of the last stored unit; the extra half step absorbs the floating-point
#: error of decoding packed values.
DEFAULT_TOLERANCE_STEPS = 1.5
