"""Command-line generator of annotated comparison reports.

Usage::

    ncjuxta-report ref.nc tgt.nc --annotations ann.json -o report.html
    ncjuxta-report ref.nc tgt.nc --annotations ann.json -o report.pdf --format pdf
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import netCDF4

from .annotated_report import ReportInputs, build_pdf, build_report
from .annotations import AnnotationError, check_against_results, load_annotations
from .comparison import FilterInfo, scan_all_groups
from .config import AppConfig, apply_config_defaults, load_config
from .constants import DEFAULT_TOLERANCE_STEPS
from .expression import create_expression_mask, find_affected_groups
from .flags import Flag

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ncjuxta-report",
        description="Generate an annotated comparison report (HTML or PDF).",
    )
    parser.add_argument("source", nargs="?", help="Source (reference) NetCDF file.")
    parser.add_argument("target", nargs="?", help="Target NetCDF file.")
    parser.add_argument("--annotations", "-a", help="Annotation JSON file.")
    parser.add_argument("--output", "-o", help="Output file (default: report.html).")
    parser.add_argument(
        "--format",
        choices=("html", "pdf"),
        help="Output format (default: guessed from the output extension).",
    )
    parser.add_argument("--config", help="TOML config file (CLI args win).")
    parser.add_argument("--expr", help="Global filter expression.")
    parser.add_argument("--flag", nargs="+", help="Flag variable path(s).")
    parser.add_argument(
        "--tolerance-steps",
        type=float,
        help="Differences up to N storage steps are 'almost equal' (default: 1.5).",
    )
    parser.add_argument(
        "--plotlyjs",
        choices=("inline", "cdn"),
        default="inline",
        help="Embed plotly.js (offline, default) or load it from the CDN.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 2 when differing variables have no annotation.",
    )
    return parser.parse_args(argv)


def prepare_inputs(
    src_ds: netCDF4.Dataset,
    tgt_ds: netCDF4.Dataset,
    src_name: str,
    tgt_name: str,
    expression: str | None = None,
    flag_paths: list[str] | None = None,
    tolerance_steps: float = DEFAULT_TOLERANCE_STEPS,
) -> ReportInputs:
    """Apply the global filter, scan every group and bundle the report inputs.

    Raises:
        ValueError: If the filter expression cannot be evaluated.

    """
    mask = None
    filter_info = None
    if expression:
        mask, error = create_expression_mask(src_ds, expression)
        if mask is None:
            raise ValueError(f"invalid expression: {error}")
        filter_info = FilterInfo(
            expression=expression,
            selected_points=int(mask.sum()),
            total_points=int(mask.size),
            affected_groups=find_affected_groups(expression),
        )
    flags = [Flag(path, tgt_ds) for path in flag_paths] if flag_paths else None
    results = scan_all_groups(
        src_ds, tgt_ds, filter_info, mask=mask, tolerance_steps=tolerance_steps
    )
    return ReportInputs(
        src_ds=src_ds,
        tgt_ds=tgt_ds,
        src_name=src_name,
        tgt_name=tgt_name,
        results=results,
        flags=flags,
        mask=mask,
        expression=expression,
        tolerance_steps=tolerance_steps,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``ncjuxta-report``."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    config = load_config(args.config) if args.config else AppConfig()
    apply_config_defaults(args, config)
    if not (args.source and args.target and args.annotations):
        print(
            "error: source, target and --annotations are required "
            "(on the command line or in --config)",
            file=sys.stderr,
        )
        return 1

    output = pathlib.Path(args.output or "report.html")
    fmt = args.format or ("pdf" if output.suffix.lower() == ".pdf" else "html")

    try:
        doc = load_annotations(args.annotations)
    except AnnotationError as exc:
        for err in exc.errors:
            print(f"error: {err}", file=sys.stderr)
        return 1

    src_path = pathlib.Path(args.source)
    tgt_path = pathlib.Path(args.target)
    with (
        netCDF4.Dataset(str(src_path), mode="r") as src_ds,
        netCDF4.Dataset(str(tgt_path), mode="r") as tgt_ds,
    ):
        try:
            inputs = prepare_inputs(
                src_ds,
                tgt_ds,
                src_path.name,
                tgt_path.name,
                args.expr,
                args.flag,
                args.tolerance_steps,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        warnings, unannotated = check_against_results(doc, inputs.results)
        for msg in warnings:
            print(f"warning: {msg}", file=sys.stderr)
        for varpath in unannotated:
            print(f"warning: no annotation for {varpath}", file=sys.stderr)
        try:
            if fmt == "pdf":
                output.write_bytes(build_pdf(doc, inputs))
            else:
                page = build_report(doc, inputs, plotlyjs=args.plotlyjs)
                output.write_text(page, encoding="utf-8")
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(f"Report written to {output}")
    if args.strict and unannotated:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
