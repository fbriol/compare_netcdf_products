"""Serialize comparison results to CSV and JSON reports."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.comparison import GroupComparisonResult


_CSV_COLUMNS = (
    "group_path",
    "varname",
    "long_name",
    "status",
    "shape_diff",
    "dtype_diff",
    "atol",
    "min_diff",
    "max_diff",
    "mean_diff",
    "std_diff",
    "message",
)


def results_to_csv(results: list[GroupComparisonResult]) -> str:
    """Render comparison results as a CSV string.

    One row per variable across every group. Stat columns are empty
    for statuses without a ``Stats`` object (equal, missing, masked).
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_COLUMNS)
    for group in results:
        for var in group.variables:
            stats = var.stats
            writer.writerow(
                [
                    group.group_path,
                    var.varname,
                    var.long_name,
                    var.status,
                    var.shape_diff,
                    var.dtype_diff,
                    var.atol,
                    f"{stats.min_diff}" if stats else "",
                    f"{stats.max_diff}" if stats else "",
                    f"{stats.mean_diff}" if stats else "",
                    f"{stats.std_diff}" if stats else "",
                    var.message,
                ]
            )
    return buf.getvalue()


def _default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    try:
        import numpy

        if isinstance(obj, numpy.generic):
            return obj.item()
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def results_to_json(results: list[GroupComparisonResult], *, indent: int = 2) -> str:
    """Render comparison results as a JSON string via ``dataclasses.asdict``."""
    payload = [dataclasses.asdict(r) for r in results]
    return json.dumps(payload, indent=indent, default=_default)
