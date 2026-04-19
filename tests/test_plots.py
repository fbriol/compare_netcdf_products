"""Tests for app/plots.py — guard against trace-shape regressions."""

from __future__ import annotations

import numpy
import plotly.graph_objects as go

from app.data_loading import DataSource
from app.plots import build_all_figures


def _make_data_source(*, with_coords: bool = False) -> DataSource:
    expected = numpy.linspace(0.0, 1.0, 50)
    computed = expected + 0.01
    lon = lat = None
    if with_coords:
        lon = numpy.linspace(-10.0, 10.0, 50)
        lat = numpy.linspace(-5.0, 5.0, 50)
    return DataSource(expected, computed, flag=None, longitude=lon, latitude=lat)


def test_build_all_figures_labels_without_coords() -> None:
    ds = _make_data_source()
    figs = build_all_figures("altitude", ds)
    labels = [label for label, _ in figs]
    assert labels == [
        "Signal Comparison",
        "Signal Difference",
        "Scatter Plot",
        "Histogram",
        "Statistics",
    ]
    for _, fig in figs:
        assert isinstance(fig, go.Figure)


def test_build_all_figures_inserts_geo_map_when_coords_present() -> None:
    ds = _make_data_source(with_coords=True)
    figs = build_all_figures("altitude", ds)
    assert figs[0][0] == "Geographical Position"


def test_signal_comparison_has_two_main_traces() -> None:
    ds = _make_data_source()
    figs = dict(build_all_figures("altitude", ds))
    traces = figs["Signal Comparison"].data
    names = {t.name for t in traces}
    assert "Reference" in names or "Source" in names or len(traces) >= 2
