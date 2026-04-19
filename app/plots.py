"""Plotly figure factories for NetCDF variable comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy
import plotly.graph_objects as go

if TYPE_CHECKING:
    from .data_loading import DataSource

# Seaborn-style palette (first colours from the Plotly "seaborn" template)
_C_SOURCE = "rgb(76,114,176)"  # muted blue
_C_TARGET = "rgb(196,78,82)"  # muted red
_C_DIFF = "rgb(85,168,104)"  # muted green
_C_HIST = "rgb(129,114,179)"  # muted purple
_C_OVERLAY = "rgb(120,120,120)"  # muted grey for secondary-axis overlay
_TEMPLATE = "seaborn"


def add_comparison_overlay(fig: go.Figure, overlay: dict[str, Any]) -> None:
    """Add an overlay variable on a secondary y-axis of a comparison figure.

    The overlay is drawn on yaxis2 so its own scale can differ from the
    primary variable's scale. Legend is enabled so the user can toggle the
    overlay. The caller is responsible for rescaling ``overlay["x"]`` into
    the primary plot's index range when sampling rates differ.

    Args:
        fig: The Signal Comparison figure to mutate.
        overlay: Dict with keys ``x``, ``y``, ``name``.

    """
    name = str(overlay.get("name", "Overlay"))
    fig.add_trace(
        go.Scattergl(
            x=overlay["x"],
            y=overlay["y"],
            mode="markers",
            marker={"size": 2, "color": _C_OVERLAY},
            name=name,
            yaxis="y2",
            hovertemplate=("index: %{x}<br>" + name + ": %{y}<extra></extra>"),
        )
    )
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "itemsizing": "constant",
        },
        yaxis2={
            "title": name,
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "fixedrange": False,
        },
        margin={"l": 60, "r": 60, "t": 10, "b": 40},
    )


def fig_signal_comparison(
    varname: str,
    data_source: DataSource,
    overlay: dict[str, Any] | None = None,
) -> go.Figure:
    """Generate a figure comparing expected and computed signals.

    Args:
        varname: The variable name.
        data_source: The data source.
        overlay: Optional overlay dict (keys ``x``, ``y``, ``name``) drawn on a
            secondary y-axis so a second variable can be visually compared.

    Returns:
        Plotly figure.

    """
    d = data_source.source_data
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=d["x"],
            y=d["expected"],
            mode="markers",
            marker={"size": 2, "color": _C_SOURCE},
            name="Source",
            hovertemplate="index: %{x}<br>source: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=d["x"],
            y=d["computed"],
            mode="markers",
            marker={"size": 2, "color": _C_TARGET},
            name="Target",
            hovertemplate="index: %{x}<br>target: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        xaxis_title="Index",
        yaxis_title="Value",
        height=500,
        margin={"l": 60, "r": 20, "t": 10, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "itemsizing": "constant",
        },
        yaxis_fixedrange=False,
    )
    if overlay is not None:
        add_comparison_overlay(fig, overlay)
    return fig


def fig_signal_difference(
    varname: str,
    data_source: DataSource,
) -> go.Figure:
    """Generate a figure of the signal difference.

    Args:
        varname: The variable name.
        data_source: The data source.

    Returns:
        Plotly figure.

    """
    d = data_source.source_data
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=d["x"],
            y=d["difference"],
            mode="markers",
            marker={"size": 2, "color": _C_DIFF},
            name="Difference",
            hovertemplate="index: %{x}<br>difference: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        xaxis_title="Index",
        yaxis_title="Source - Target",
        yaxis_fixedrange=False,
        height=400,
        margin={"l": 60, "r": 20, "t": 10, "b": 40},
        showlegend=False,
    )
    return fig


def fig_nan_distribution(
    varname: str,
    data_source: DataSource,
) -> go.Figure:
    """Generate a figure showing exclusive NaN positions.

    Args:
        varname: The variable name.
        data_source: The data source.

    Returns:
        Plotly figure.

    """
    d = data_source.source_data
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=d["x"],
            y=d["nan_expected"],
            mode="markers",
            marker={"size": 3, "color": _C_SOURCE, "symbol": "square"},
            name="Only in Source",
            hovertemplate="index: %{x}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=d["x"],
            y=d["nan_computed"],
            mode="markers",
            marker={"size": 3, "color": _C_TARGET, "symbol": "square"},
            name="Only in Target",
            hovertemplate="index: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        xaxis_title="Index",
        yaxis_title="Value",
        yaxis_fixedrange=False,
        height=400,
        margin={"l": 60, "r": 20, "t": 10, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "itemsizing": "constant",
        },
    )
    return fig


def fig_scatter(
    varname: str,
    data_source: DataSource,
) -> go.Figure:
    """Generate a scatter plot of computed vs expected.

    Args:
        varname: The variable name.
        data_source: The data source.

    Returns:
        Plotly figure.

    """
    d = data_source.source_data
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=d["computed"],
            y=d["expected"],
            mode="markers",
            marker={"size": 2},
            name="",
            hovertemplate="target: %{x}<br>source: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        xaxis_title="Target",
        yaxis_title="Source",
        height=500,
        margin={"l": 60, "r": 20, "t": 10, "b": 40},
    )
    return fig


def fig_histogram(
    varname: str,
    data_source: DataSource,
) -> go.Figure:
    """Generate a histogram of differences.

    Args:
        varname: The variable name.
        data_source: The data source.

    Returns:
        Plotly figure.

    """
    h = data_source.histogram_data
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[(lt + rt) / 2 for lt, rt in zip(h["left"], h["right"], strict=False)],
            y=h["top"],
            width=[rt - lt for lt, rt in zip(h["left"], h["right"], strict=False)],
            marker_color=_C_HIST,
            hovertemplate="range: %{x}<br>count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        xaxis_title="Difference",
        yaxis_title="Frequency",
        height=400,
        margin={"l": 60, "r": 20, "t": 10, "b": 40},
        bargap=0.02,
    )
    return fig


def fig_stats_table(
    data_source: DataSource,
) -> go.Figure:
    """Generate a statistics table as a Plotly Table figure.

    Args:
        data_source: The data source.

    Returns:
        Plotly figure with a table trace.

    """
    s = data_source.stats_data
    values_formatted = []
    for v in s["value"]:
        if isinstance(v, (int, numpy.integer)):
            values_formatted.append(f"{v:,}")
        else:
            values_formatted.append(f"{v:.6g}")

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["<b>Metric</b>", "<b>Value</b>"],
                    "fill_color": "rgb(76,114,176)",
                    "font": {"color": "white", "size": 12},
                    "align": "left",
                },
                cells={
                    "values": [s["metric"], values_formatted],
                    "fill_color": [["rgb(234,234,242)", "white"] * 3],
                    "align": "left",
                    "font": {"size": 12, "color": "rgb(36,36,36)"},
                    "height": 28,
                },
            )
        ]
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=240,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    return fig


def fig_geo_map(
    varname: str,
    data_source: DataSource,
) -> go.Figure:
    """Generate a geographical map showing measurement positions.

    Two traces (Expected / Computed) are plotted on a world map with
    coastlines.  Each point is coloured by the variable value using a
    shared colour scale so the user can compare values spatially.

    Args:
        varname: The variable name.
        data_source: The data source (must contain longitude/latitude).

    Returns:
        Plotly figure.

    """
    d = data_source.source_data
    lon = numpy.asarray(d["longitude"], dtype=float)
    lat = numpy.asarray(d["latitude"], dtype=float)
    expected = numpy.asarray(d["expected"], dtype=float)
    computed = numpy.asarray(d["computed"], dtype=float)

    # Aggressive downsampling for the map (Scattergeo is slow)
    _MAX_MAP_POINTS = 5_000
    n = len(lon)
    if n > _MAX_MAP_POINTS:
        step = max(1, n // _MAX_MAP_POINTS)
        idx = numpy.arange(0, n, step)
        lon = lon[idx]
        lat = lat[idx]
        expected = expected[idx]
        computed = computed[idx]

    # Remove NaN positions (no coordinates)
    valid_pos = ~(numpy.isnan(lon) | numpy.isnan(lat))
    lon = lon[valid_pos]
    lat = lat[valid_pos]
    expected = expected[valid_pos]
    computed = computed[valid_pos]

    # Shared colour range
    all_vals = numpy.concatenate([expected, computed])
    valid_vals = all_vals[~numpy.isnan(all_vals)]
    if len(valid_vals) > 0:
        cmin, cmax = float(numpy.nanmin(valid_vals)), float(numpy.nanmax(valid_vals))
    else:
        cmin, cmax = 0.0, 1.0

    fig = go.Figure()

    # --- Reference trace ---
    fig.add_trace(
        go.Scattergeo(
            lon=lon,
            lat=lat,
            mode="markers",
            marker={
                "size": 4,
                "color": expected,
                "colorscale": "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "showscale": False,
            },
            name="Source",
            hovertemplate=(
                "lon: %{lon:.3f}<br>lat: %{lat:.3f}"
                "<br>source: %{marker.color:.4g}<extra></extra>"
            ),
        )
    )

    # --- Target trace ---
    fig.add_trace(
        go.Scattergeo(
            lon=lon,
            lat=lat,
            mode="markers",
            marker={
                "size": 4,
                "color": computed,
                "colorscale": "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "showscale": False,
            },
            name="Target",
            visible="legendonly",
            hovertemplate=(
                "lon: %{lon:.3f}<br>lat: %{lat:.3f}"
                "<br>target: %{marker.color:.4g}<extra></extra>"
            ),
        )
    )

    # Auto-fit the map to the data points
    valid_lon = lon[~numpy.isnan(lon)]
    valid_lat = lat[~numpy.isnan(lat)]
    geo_kwargs = {
        "showcoastlines": True,
        "coastlinecolor": "#555",
        "showland": True,
        "landcolor": "#e8e8e8",
        "showocean": True,
        "oceancolor": "#d4ecf7",
        "showlakes": True,
        "lakecolor": "#d4ecf7",
        "showcountries": True,
        "countrycolor": "#aaa",
        "projection_type": "natural earth",
    }
    if len(valid_lon) > 0:
        margin_deg = 2.0
        geo_kwargs["lonaxis_range"] = [
            float(valid_lon.min()) - margin_deg,
            float(valid_lon.max()) + margin_deg,
        ]
        geo_kwargs["lataxis_range"] = [
            float(valid_lat.min()) - margin_deg,
            float(valid_lat.max()) + margin_deg,
        ]
    fig.update_geos(**geo_kwargs)

    fig.update_layout(
        template=_TEMPLATE,
        height=500,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "itemsizing": "constant",
        },
    )
    return fig


def build_all_figures(
    varname: str,
    data_source: DataSource,
    comparison_overlay: dict[str, Any] | None = None,
) -> list[tuple[str, go.Figure]]:
    """Build all comparison figures for a variable.

    Args:
        varname: The variable name for titles.
        data_source: The DataSource containing all data.
        comparison_overlay: Optional overlay (``x``/``y``/``name``) rendered on
            the Signal Comparison plot's secondary y-axis.

    Returns:
        List of (label, figure) tuples.

    """
    plot_configs: list[tuple[str, go.Figure]] = [
        (
            "Signal Comparison",
            fig_signal_comparison(varname, data_source, overlay=comparison_overlay),
        ),
        ("Signal Difference", fig_signal_difference(varname, data_source)),
        ("Scatter Plot", fig_scatter(varname, data_source)),
        ("Histogram", fig_histogram(varname, data_source)),
        ("Statistics", fig_stats_table(data_source)),
    ]

    if data_source.exclusive_nan:
        plot_configs.insert(
            -1,
            (
                "Exclusive Undefined Positions",
                fig_nan_distribution(varname, data_source),
            ),
        )

    # Add geographical map if coordinates are available
    if "longitude" in data_source.source_data:
        plot_configs.insert(
            0,
            ("Geographical Position", fig_geo_map(varname, data_source)),
        )

    return plot_configs
