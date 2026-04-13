"""Panel application entry point for NetCDF comparison.

Usage:
    panel serve app/main.py --args ref.nc tgt.nc \
        [--flag VAR_PATH ...] [--expr EXPRESSION]

"""

from __future__ import annotations

import argparse
import logging
import pathlib

import netCDF4
import panel as pn

from .comparison import FilterInfo, scan_all_groups
from .expression import create_expression_mask, find_target_group
from .flags import Flag
from .state import SessionState
from .widgets import build_expression_builder, build_group_tabs, build_legend

LOGGER = logging.getLogger(__name__)

pn.extension("tabulator", sizing_mode="stretch_width")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed via --args."""
    parser = argparse.ArgumentParser(
        description="Interactive NetCDF comparison application."
    )
    parser.add_argument(
        "reference", type=str, help="Path to the reference NetCDF file."
    )
    parser.add_argument("target", type=str, help="Path to the target NetCDF file.")
    parser.add_argument(
        "--flag",
        type=str,
        nargs="+",
        help="Variable path(s) for flag-based filtering. "
        "The variable must have flag_meanings and flag_values attributes.",
    )
    parser.add_argument(
        "--expr",
        type=str,
        help="Initial expression to filter data. "
        "Example: (ds['/data_20/distance_to_coast'][:] > 30000)",
    )
    return parser.parse_args()


def create_app() -> pn.template.FastListTemplate:
    """Create the Panel application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()

    ref_path = pathlib.Path(args.reference).resolve()
    tgt_path = pathlib.Path(args.target).resolve()

    LOGGER.info("Opening reference: %s", ref_path)
    LOGGER.info("Opening target: %s", tgt_path)

    ref_ds = netCDF4.Dataset(str(ref_path), mode="r")
    tgt_ds = netCDF4.Dataset(str(tgt_path), mode="r")

    # Build flags
    flags = None
    if args.flag:
        flags = [Flag(varname, tgt_ds) for varname in args.flag]

    # Build initial expression mask
    active_mask = None
    active_expression = args.expr
    filter_info = None

    if active_expression:
        LOGGER.info("Applying initial expression: %s", active_expression)
        mask, error = create_expression_mask(ref_ds, active_expression)
        if mask is not None:
            active_mask = mask
            _, group_name, _ = find_target_group(ref_ds, active_expression)
            if group_name:
                selected = int(mask.sum())
                total = int(mask.size)
                filter_info = FilterInfo(
                    expression=active_expression,
                    selected_points=selected,
                    total_points=total,
                    affected_groups={group_name},
                )
        else:
            LOGGER.error("Expression error: %s", error)

    # Create session state
    state = SessionState(
        ref_path=ref_path,
        tgt_path=tgt_path,
        ref_ds=ref_ds,
        tgt_ds=tgt_ds,
        flags=flags,
        active_mask=active_mask,
        active_expression=active_expression,
        filter_info=filter_info,
    )

    # Initial comparison scan
    LOGGER.info("Scanning groups...")
    state.group_results = scan_all_groups(ref_ds, tgt_ds, filter_info, mask=active_mask)

    # Build the main content area (will be rebuilt on expression changes)
    main_content = pn.Column(sizing_mode="stretch_both")

    def refresh_main_content() -> None:
        """Rebuild the main content after expression/filter changes."""
        main_content.clear()
        tabs = build_group_tabs(state)
        main_content.append(tabs)

    refresh_main_content()

    # Build sidebar
    sidebar = [
        pn.pane.Markdown(f"**Reference:**\n`{ref_path.name}`"),
        pn.pane.Markdown(f"**Target:**\n`{tgt_path.name}`"),
        pn.layout.Divider(),
        build_expression_builder(state, refresh_main_content),
        pn.layout.Divider(),
        build_legend(),
    ]

    # Build template
    template = pn.template.FastListTemplate(
        title=f"NetCDF Comparison: {ref_path.name} vs {tgt_path.name}",
        sidebar=sidebar,
        main=[main_content],
        sidebar_width=380,
        accent_base_color="#2196F3",
        header_background="#2196F3",
    )

    return template


# Panel serve entry point
template = create_app()
template.servable()
