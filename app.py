#!/usr/bin/env python
"""Dash application entry point for NetCDF comparison.

Usage:
    python app.py ref.nc tgt.nc [--flag VAR_PATH ...] [--expr EXPRESSION]
    python app.py ref.nc tgt.nc --port 8050

"""

from __future__ import annotations

import argparse
import logging
import pathlib

import dash
import netCDF4

from app.callbacks import register_callbacks
from app.comparison import FilterInfo, scan_all_groups
from app.expression import (
    create_expression_mask,
    find_affected_groups,
    rewrite_expression,
)
from app.flags import Flag
from app.layout import create_layout
from app.state import SessionState

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive NetCDF comparison application (Dash)."
    )
    parser.add_argument("source", type=str, help="Path to the source NetCDF file.")
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
        "Example: ${/data_20/distance_to_coast} > 30000",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port to serve the application on (default: 5006).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    """Create and run the Dash application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()

    src_path = pathlib.Path(args.source).resolve()
    tgt_path = pathlib.Path(args.target).resolve()

    LOGGER.info("Opening source: %s", src_path)
    LOGGER.info("Opening target: %s", tgt_path)

    src_ds = netCDF4.Dataset(str(src_path), mode="r")
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
        rewritten = rewrite_expression(active_expression)
        mask, error = create_expression_mask(src_ds, rewritten)
        if mask is not None:
            active_mask = mask
            affected = find_affected_groups(active_expression)
            selected = int(mask.sum())
            total = int(mask.size)
            filter_info = FilterInfo(
                expression=active_expression,
                selected_points=selected,
                total_points=total,
                affected_groups=affected,
            )
        else:
            LOGGER.error("Expression error: %s", error)

    # Create session state
    state = SessionState(
        src_path=src_path,
        tgt_path=tgt_path,
        src_ds=src_ds,
        tgt_ds=tgt_ds,
        flags=flags,
        active_mask=active_mask,
        active_expression=active_expression,
        filter_info=filter_info,
    )

    # Initial comparison scan
    LOGGER.info("Scanning groups...")
    state.group_results = scan_all_groups(src_ds, tgt_ds, filter_info, mask=active_mask)

    # Create Dash app
    dash_app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="NetCDF Comparison",
    )
    dash_app.layout = create_layout(state)
    register_callbacks(dash_app, state)

    LOGGER.info("Starting Dash server on port %d", args.port)
    dash_app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
