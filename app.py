#!/usr/bin/env python
"""Dash application entry point for NetCDF comparison.

Usage:
    python app.py ref.nc tgt.nc [--flag VAR_PATH ...] [--expr EXPRESSION]
    python app.py ref.nc tgt.nc --port 8050

"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import secrets

import dash
import netCDF4

from app.annotations import AnnotationError, AnnotationStore
from app.callbacks import register_callbacks
from app.comparison import FilterInfo, scan_all_groups
from app.config import AppConfig, apply_config_defaults, load_config
from app.expression import (
    create_expression_mask,
    find_affected_groups,
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
    parser.add_argument(
        "source", type=str, nargs="?", help="Path to the source NetCDF file."
    )
    parser.add_argument(
        "target", type=str, nargs="?", help="Path to the target NetCDF file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML config file. CLI args override config values.",
    )
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
        default=None,
        help="Port to serve the application on (default: 5006).",
    )
    parser.add_argument(
        "--tolerance-steps",
        type=float,
        default=None,
        help="Differences up to N storage steps (scale_factor) are classified "
        "'almost equal' (default: 1.5).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Annotation JSON file to display and edit (created if missing).",
    )
    parser.add_argument(
        "--annotations-readonly",
        action="store_true",
        help="Display the annotations without the editing controls.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )
    return parser.parse_args()


def build_session_state(
    args: argparse.Namespace,
    src_path: pathlib.Path,
    tgt_path: pathlib.Path,
    src_ds: netCDF4.Dataset,
    tgt_ds: netCDF4.Dataset,
) -> SessionState:
    """Build the initial session state for the Dash application."""
    flags = None
    if args.flag:
        flags = [Flag(varname, tgt_ds) for varname in args.flag]

    active_mask = None
    active_expression = args.expr
    filter_info = None

    if active_expression:
        LOGGER.info("Applying initial expression: %s", active_expression)
        mask, error = create_expression_mask(src_ds, active_expression)
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

    annotations = None
    if args.annotations:
        LOGGER.info("Loading annotations: %s", args.annotations)
        try:
            annotations = AnnotationStore(
                args.annotations, readonly=args.annotations_readonly
            )
        except (AnnotationError, FileNotFoundError) as exc:
            raise SystemExit(f"Cannot load annotations: {exc}") from exc

    # Create the session state object
    state = SessionState(
        src_path=src_path,
        tgt_path=tgt_path,
        src_ds=src_ds,
        tgt_ds=tgt_ds,
        flags=flags,
        active_mask=active_mask,
        active_expression=active_expression,
        filter_info=filter_info,
        annotations=annotations,
        tolerance_steps=args.tolerance_steps,
    )

    # Initial scan of groups to populate the state with comparison results
    LOGGER.info("Scanning groups...")
    state.group_results = scan_all_groups(
        src_ds,
        tgt_ds,
        filter_info,
        mask=active_mask,
        tolerance_steps=args.tolerance_steps,
    )
    return state


def main() -> None:
    """Create and run the Dash application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()

    config = load_config(args.config) if args.config else AppConfig()
    apply_config_defaults(args, config)

    if args.source is None or args.target is None:
        raise SystemExit(
            "Source and target NetCDF paths are required "
            "(via CLI positional args or [paths] in --config)."
        )

    logging.getLogger().setLevel(config.logging.level.upper())

    src_path = pathlib.Path(args.source).resolve()
    tgt_path = pathlib.Path(args.target).resolve()

    LOGGER.info("Opening source: %s", src_path)
    LOGGER.info("Opening target: %s", tgt_path)

    src_ds = netCDF4.Dataset(str(src_path), mode="r")
    tgt_ds = netCDF4.Dataset(str(tgt_path), mode="r")

    state: SessionState | None = None
    try:
        state = build_session_state(args, src_path, tgt_path, src_ds, tgt_ds)

        # Create Dash app
        dash_app = dash.Dash(
            __name__,
            suppress_callback_exceptions=True,
            title="NetCDF Comparison",
        )
        # Flask session (backing per-tab SessionData) requires a secret key.
        # Respect an operator-supplied key if present; otherwise generate
        # one per-process so cookies survive until restart.
        dash_app.server.secret_key = os.environ.get(
            "NCJUXTA_SECRET_KEY", secrets.token_hex(32)
        )
        dash_app.layout = create_layout(state)
        register_callbacks(dash_app, state)

        LOGGER.info("Starting Dash server on port %d", args.port)
        dash_app.run(debug=args.debug, port=args.port)
    finally:
        if state is not None:
            state.close()
        else:
            src_ds.close()
            tgt_ds.close()


if __name__ == "__main__":
    main()
