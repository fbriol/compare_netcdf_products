"""View builder functions for the Dash application.

Builds the HTML content for each view (group browser, detail view,
global summary, breadcrumb, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy
import plotly.graph_objects as go
from dash import dcc, html

if TYPE_CHECKING:
    pass

from ..constants import STATUS_COLORS, STATUS_TEXT_COLORS
from ..data_loading import load_variable_data
from ..plots import build_all_figures
from ..utils import (
    count_diffs_recursive,
    find_group_result,
    navigate_to_group,
)

LOGGER = logging.getLogger(__name__)


def build_breadcrumb(group_path: str) -> list[Any]:
    """Build clickable breadcrumb components."""
    parts = [p for p in group_path.split("/") if p]
    items: list[Any] = []

    items.append(
        html.Button(
            "\U0001f3e0",
            id={"type": "breadcrumb-btn", "path": "/"},
            n_clicks=0,
            style={
                "border": "none",
                "background": "transparent",
                "cursor": "pointer",
                "fontSize": "16px",
                "padding": "4px 8px",
            },
        )
    )

    accumulated = ""
    for i, part in enumerate(parts):
        accumulated += f"/{part}"
        items.append(
            html.Span(
                "\u203a",
                style={
                    "color": "#90a4ae",
                    "margin": "0 4px",
                    "fontSize": "16px",
                },
            )
        )
        is_last = i == len(parts) - 1
        items.append(
            html.Button(
                part,
                id={"type": "breadcrumb-btn", "path": accumulated},
                n_clicks=0,
                style={
                    "border": "none",
                    "background": "transparent",
                    "cursor": "pointer",
                    "fontSize": "13px",
                    "padding": "4px 8px",
                    "fontWeight": "600" if is_last else "400",
                    "color": "#263238" if is_last else "#607d8b",
                },
            )
        )

    return items


def get_clickable_varpaths(state):
    """Build a flat list of variable paths for prev/next navigation.

    Only variables with differences are included in the navigation
    sequence.  Other variables can still be viewed by clicking in
    the group browser.
    """
    paths: list[str] = []
    for gr in state.group_results:
        paths.extend(
            v.varpath for v in gr.variables if v.status in ("different", "almost_equal")
        )
    return paths


def build_global_summary(state):
    """Build a summary table of ALL problematic variables across all groups."""
    problem_vars: list[Any] = []
    for gr in state.group_results:
        problem_vars.extend(
            v
            for v in gr.variables
            if v.status in ("different", "almost_equal") and v.stats is not None
        )
    if not problem_vars:
        return []

    # Sort by max absolute error descending
    problem_vars.sort(key=lambda v: abs(float(v.stats.max_diff)), reverse=True)

    content: list[Any] = []
    content.append(
        html.H4(
            "\u26a0 Global Summary \u2014 Variables with Differences",
            style={"color": "#c62828", "margin": "16px 0 8px 0"},
        )
    )
    content.append(
        html.Div(
            [
                html.Span(
                    f"{len(problem_vars)} variable(s) with differences across all "
                    "groups. Sorted by max |diff| descending. Click a row to inspect.",
                    style={"fontSize": "11px", "color": "#78909c"},
                ),
                dcc.Input(
                    id="summary-search-input",
                    type="text",
                    placeholder="Filter variables\u2026",
                    debounce=False,
                    list="summary-search-options",
                    style={
                        "marginLeft": "16px",
                        "padding": "4px 10px",
                        "border": "1px solid #cfd8dc",
                        "borderRadius": "4px",
                        "fontSize": "12px",
                        "width": "250px",
                    },
                ),
                html.Datalist(
                    id="summary-search-options",
                    children=[html.Option(value=v.varpath) for v in problem_vars],
                ),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
        )
    )

    rows = []
    for v in problem_vars:
        bg = STATUS_COLORS.get(v.status, "#ffffff")
        fg = STATUS_TEXT_COLORS.get(v.status, "#000000")
        cell_style = {"padding": "6px 12px", "fontSize": "12px"}
        rows.append(
            html.Tr(
                [
                    html.Td(v.varpath, style=cell_style),
                    html.Td(v.status.replace("_", " ").title(), style=cell_style),
                    html.Td(f"{abs(float(v.stats.max_diff)):.6g}", style=cell_style),
                    html.Td(f"{abs(float(v.stats.mean_diff)):.6g}", style=cell_style),
                    html.Td(f"{float(v.stats.std_diff):.6g}", style=cell_style),
                ],
                id={"type": "summary-var-row", "path": v.varpath},
                n_clicks=0,
                style={
                    "backgroundColor": bg,
                    "color": fg,
                    "cursor": "pointer",
                },
            )
        )

    content.append(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(
                                "Variable Path",
                                style={"padding": "8px 12px", "textAlign": "left"},
                            ),
                            html.Th(
                                "Status",
                                style={"padding": "8px 12px", "textAlign": "left"},
                            ),
                            html.Th(
                                "Max |diff|",
                                style={"padding": "8px 12px", "textAlign": "left"},
                            ),
                            html.Th(
                                "Mean |diff|",
                                style={"padding": "8px 12px", "textAlign": "left"},
                            ),
                            html.Th(
                                "Std",
                                style={"padding": "8px 12px", "textAlign": "left"},
                            ),
                        ],
                        style={"backgroundColor": "rgb(76,114,176)", "color": "white"},
                    )
                ),
                html.Tbody(rows, id="summary-table-body"),
            ],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "border": "1px solid #e0e0e0",
                "fontSize": "12px",
                "marginBottom": "20px",
            },
        )
    )

    return content


def build_group_browser(state, group_path):
    """Build group browser view content."""
    gr = find_group_result(state, group_path)
    content: list[Any] = []

    # At root: show a toggle between Summary and Browse views
    if group_path == "/":
        summary_items = build_global_summary(state)
        if summary_items:
            content.append(
                html.Div(
                    [
                        html.Button(
                            "\u26a0 Summary",
                            id="root-tab-summary",
                            n_clicks=0,
                            style={
                                "border": "none",
                                "borderBottom": "3px solid rgb(76,114,176)",
                                "padding": "8px 20px",
                                "cursor": "pointer",
                                "fontSize": "13px",
                                "fontWeight": "600",
                                "backgroundColor": "transparent",
                                "color": "rgb(76,114,176)",
                            },
                        ),
                        html.Button(
                            "\U0001f4c2 Browse groups",
                            id="root-tab-browse",
                            n_clicks=0,
                            style={
                                "border": "none",
                                "borderBottom": "3px solid transparent",
                                "padding": "8px 20px",
                                "cursor": "pointer",
                                "fontSize": "13px",
                                "fontWeight": "400",
                                "backgroundColor": "transparent",
                                "color": "#78909c",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "4px",
                        "borderBottom": "1px solid #e0e0e0",
                        "marginBottom": "12px",
                    },
                )
            )
            # Summary panel (visible by default)
            content.append(
                html.Div(
                    summary_items,
                    id="root-panel-summary",
                    style={"display": "block"},
                )
            )
    if gr is None:
        content.append(
            html.Div(
                f"Group '{group_path}' not found in comparison results.",
                style={
                    "padding": "12px",
                    "backgroundColor": "#fff3cd",
                    "borderRadius": "4px",
                    "color": "#856404",
                },
            )
        )
        return content

    # Summary bar
    n_diff = sum(1 for v in gr.variables if v.status in ("different", "almost_equal"))
    n_eq = sum(1 for v in gr.variables if v.status == "equal")
    n_aeq = sum(1 for v in gr.variables if v.status == "almost_equal")
    n_masked = sum(1 for v in gr.variables if v.status == "masked")
    n_miss_tgt = sum(1 for v in gr.variables if v.status == "missing_in_target")
    n_miss_src = sum(1 for v in gr.variables if v.status == "missing_in_source")

    summary_parts = []
    if n_eq:
        summary_parts.append(
            html.Span(f"\u2713 {n_eq} equal", style={"color": "#2e7d32"})
        )
    if n_aeq:
        summary_parts.append(
            html.Span(f"\u2248 {n_aeq} almost equal", style={"color": "#006400"})
        )
    if n_diff:
        summary_parts.append(
            html.Span(f"\u2260 {n_diff} different", style={"color": "#e65100"})
        )
    if n_masked:
        summary_parts.append(
            html.Span(f"\u2014 {n_masked} masked", style={"color": "#757575"})
        )
    if n_miss_tgt:
        summary_parts.append(
            html.Span(
                f"\u2716 {n_miss_tgt} missing in target",
                style={"color": "#c62828"},
            )
        )
    if n_miss_src:
        summary_parts.append(
            html.Span(
                f"\u2716 {n_miss_src} missing in source",
                style={"color": "#3949ab"},
            )
        )

    if summary_parts:
        separator = html.Span(" \u2022 ", style={"color": "#bbb"})
        parts_with_sep = []
        for i, sp in enumerate(summary_parts):
            if i > 0:
                parts_with_sep.append(separator)
            parts_with_sep.append(sp)

        content.append(
            html.Div(
                [
                    html.B(f"{len(gr.variables)}"),
                    " variables: ",
                    *parts_with_sep,
                ],
                style={
                    "fontSize": "13px",
                    "padding": "8px 12px",
                    "background": "#eaeaf2",
                    "borderRadius": "6px",
                    "marginBottom": "12px",
                },
            )
        )

    # Filter info
    if gr.filter_description:
        content.append(
            html.Div(
                f"\U0001f50d {gr.filter_description}",
                style={
                    "padding": "8px 12px",
                    "backgroundColor": "rgb(234,234,242)",
                    "borderRadius": "4px",
                    "color": "rgb(76,114,176)",
                    "fontSize": "12px",
                    "marginBottom": "12px",
                },
            )
        )

    # Sub-groups as buttons
    ref_group = navigate_to_group(state.src_ds, group_path)
    if ref_group is not None and ref_group.groups:
        content.append(
            html.H4(
                "\U0001f4c2 Sub-groups",
                style={"color": "#37474f", "margin": "16px 0 8px 0"},
            )
        )
        btn_row = []
        for grp_name in sorted(ref_group.groups):
            child_path = (
                f"{group_path}/{grp_name}" if group_path != "/" else f"/{grp_name}"
            )
            n_d, _n_t = count_diffs_recursive(state, state.src_ds, child_path)
            label = f"\U0001f4c1 {grp_name}"
            if n_d > 0:
                label += f" ({n_d} diffs)"
            bg_color = "#fff3cd" if n_d > 0 else "#eaeaf2"
            border_color = "#ffc107" if n_d > 0 else "#e0e0e0"
            btn_row.append(
                html.Button(
                    label,
                    id={"type": "group-btn", "path": child_path},
                    n_clicks=0,
                    style={
                        "border": f"1px solid {border_color}",
                        "borderRadius": "4px",
                        "padding": "6px 12px",
                        "backgroundColor": bg_color,
                        "cursor": "pointer",
                        "fontSize": "12px",
                        "marginRight": "6px",
                        "marginBottom": "6px",
                        "textAlign": "left",
                    },
                )
            )
        content.append(html.Div(btn_row, style={"display": "flex", "flexWrap": "wrap"}))

    # Attributes table
    if gr.attributes:
        content.append(
            html.H4(
                "\U0001f4cb Attributes",
                style={"color": "#37474f", "margin": "16px 0 8px 0"},
            )
        )
        rows = []
        rows.extend(
            [
                html.Tr(
                    [
                        html.Td(a.name, style={"padding": "6px 12px"}),
                        html.Td(str(a.src_value), style={"padding": "6px 12px"}),
                        html.Td(str(a.tgt_value), style={"padding": "6px 12px"}),
                    ]
                )
                for a in gr.attributes
            ]
        )
        content.append(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(
                                    "Attribute",
                                    style={"padding": "8px 12px", "textAlign": "left"},
                                ),
                                html.Th(
                                    "Source Value",
                                    style={"padding": "8px 12px", "textAlign": "left"},
                                ),
                                html.Th(
                                    "Target Value",
                                    style={"padding": "8px 12px", "textAlign": "left"},
                                ),
                            ],
                            style={"backgroundColor": "#eaeaf2"},
                        )
                    ),
                    html.Tbody(rows),
                ],
                style={
                    "width": "100%",
                    "borderCollapse": "collapse",
                    "border": "1px solid #e0e0e0",
                    "marginBottom": "12px",
                    "fontSize": "12px",
                },
            )
        )

    # Variables table
    if gr.variables:
        content.append(
            html.H4(
                "\U0001f4ca Variables",
                style={"color": "#37474f", "margin": "16px 0 8px 0"},
            )
        )
        if n_diff > 0:
            content.append(
                html.Div(
                    "Click a coloured row to see detailed comparison plots",
                    style={
                        "fontSize": "11px",
                        "color": "#78909c",
                        "marginBottom": "6px",
                    },
                )
            )

        # Search field for filtering variables (with autocomplete)
        var_names = [v.varpath.rsplit("/", 1)[-1] for v in gr.variables]
        content.append(
            html.Div(
                [
                    dcc.Input(
                        id="var-search-input",
                        type="text",
                        placeholder="Search variables...",
                        debounce=False,
                        list="var-search-options",
                        style={
                            "width": "300px",
                            "padding": "6px 12px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "4px",
                            "fontSize": "12px",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Datalist(
                        id="var-search-options",
                        children=[html.Option(value=n) for n in sorted(set(var_names))],
                    ),
                ]
            )
        )

        var_rows = []
        for v in gr.variables:
            value_info = v.message or v.status.replace("_", " ").title()
            if v.stats is not None:
                value_info = f"{v.stats}, atol={v.atol}"
            short_name = v.varpath.rsplit("/", 1)[-1]
            bg = STATUS_COLORS.get(v.status, "#ffffff")
            fg = STATUS_TEXT_COLORS.get(v.status, "#000000")
            row_style = {
                "backgroundColor": bg,
                "color": fg,
                "cursor": "pointer",
            }
            cell_style = {"padding": "6px 12px", "fontSize": "12px"}
            var_rows.append(
                html.Tr(
                    [
                        html.Td(short_name, style=cell_style),
                        html.Td(v.status.replace("_", " ").title(), style=cell_style),
                        html.Td(
                            "\u2260" if v.shape_diff else "\u2713", style=cell_style
                        ),
                        html.Td(
                            "\u2260" if v.dtype_diff else "\u2713", style=cell_style
                        ),
                        html.Td(
                            value_info,
                            style={
                                **cell_style,
                                "maxWidth": "300px",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                            },
                        ),
                    ],
                    id={"type": "var-row", "path": v.varpath},
                    n_clicks=0,
                    style=row_style,
                )
            )

        content.append(
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th(
                                        "Variable",
                                        style={
                                            "padding": "8px 12px",
                                            "textAlign": "left",
                                        },
                                    ),
                                    html.Th(
                                        "Status",
                                        style={
                                            "padding": "8px 12px",
                                            "textAlign": "left",
                                        },
                                    ),
                                    html.Th(
                                        "Shape",
                                        style={
                                            "padding": "8px 12px",
                                            "textAlign": "left",
                                        },
                                    ),
                                    html.Th(
                                        "DType",
                                        style={
                                            "padding": "8px 12px",
                                            "textAlign": "left",
                                        },
                                    ),
                                    html.Th(
                                        "Values",
                                        style={
                                            "padding": "8px 12px",
                                            "textAlign": "left",
                                        },
                                    ),
                                ],
                                style={
                                    "backgroundColor": "rgb(76,114,176)",
                                    "color": "white",
                                },
                            )
                        ),
                        html.Tbody(var_rows, id="var-table-body"),
                    ],
                    style={
                        "width": "100%",
                        "borderCollapse": "collapse",
                        "border": "1px solid #e0e0e0",
                        "fontSize": "12px",
                    },
                ),
                id="var-table-container",
            )
        )

    # At root with summary: wrap group browser items into the Browse panel
    if group_path == "/" and any(
        getattr(c, "id", None) == "root-panel-summary" for c in content
    ):
        # Split content: tabs + summary panel vs group browser items
        tabs_and_summary = []
        browse_items = []
        found_summary = False
        for c in content:
            cid = getattr(c, "id", None)
            if cid == "root-panel-summary":
                found_summary = True
                tabs_and_summary.append(c)
            elif not found_summary:
                tabs_and_summary.append(c)
            else:
                browse_items.append(c)
        # Insert browse panel after summary panel
        tabs_and_summary.append(
            html.Div(
                browse_items,
                id="root-panel-browse",
                style={"display": "none"},
            )
        )
        return tabs_and_summary

    return content


def get_map_arrays(data_source):
    """Return downsampled arrays for the map, cached on the DataSource."""
    if hasattr(data_source, "_map_cache"):
        return data_source._map_cache

    d = data_source.source_data
    if "longitude" not in d:
        data_source._map_cache = None
        return None

    lon = numpy.asarray(d["longitude"], dtype=float)
    lat = numpy.asarray(d["latitude"], dtype=float)
    expected = numpy.asarray(d["expected"], dtype=float)
    computed = numpy.asarray(d["computed"], dtype=float)
    x = numpy.asarray(d["x"], dtype=float)

    # Downsample for map performance (Scattergeo is slow)
    _MAX_MAP = 5_000
    n = len(lon)
    if n > _MAX_MAP:
        step = max(1, n // _MAX_MAP)
        idx = numpy.arange(0, n, step)
        lon, lat = lon[idx], lat[idx]
        expected, computed, x = expected[idx], computed[idx], x[idx]

    all_valid = ~(numpy.isnan(lon) | numpy.isnan(lat))
    result = {
        "lon": lon,
        "lat": lat,
        "expected": expected,
        "computed": computed,
        "x": x,
        "all_valid": all_valid,
        "all_lon": lon[all_valid],
        "all_lat": lat[all_valid],
    }
    data_source._map_cache = result
    return result


def build_filtered_map(data_source, zoom_range):
    """Build a Scattergeo figure filtered by zoom_range.

    When a zoom is active, full-resolution data within the zoom window
    is used (downsampled to _MAX_MAP points) so the map stays consistent
    with the 1-D plots.
    """
    _MAX_MAP = 5_000

    has_zoom = zoom_range and not zoom_range.get("auto")

    if has_zoom:
        # Use full-resolution data within zoom window
        if (
            not hasattr(data_source, "full_longitude")
            or data_source.full_longitude is None
        ):
            return None
        xmin = float(zoom_range["min"])
        xmax = float(zoom_range["max"])
        x_full = data_source.full_x
        win = (x_full >= xmin) & (x_full <= xmax)
        x = x_full[win]
        lon = numpy.asarray(data_source.full_longitude[win], dtype=float)
        lat = numpy.asarray(data_source.full_latitude[win], dtype=float)
        expected = data_source.full_expected[win]
        computed = data_source.full_computed[win]
        # Downsample for map performance
        n = len(lon)
        if n > _MAX_MAP:
            step = max(1, n // _MAX_MAP)
            idx = numpy.arange(0, n, step)
            lon, lat = lon[idx], lat[idx]
            expected, computed, x = expected[idx], computed[idx], x[idx]
        all_valid = ~(numpy.isnan(lon) | numpy.isnan(lat))
        mask = all_valid
        # Background track from global cache
        bg_arrays = get_map_arrays(data_source)
        all_lon = bg_arrays["all_lon"] if bg_arrays else lon[all_valid]
        all_lat = bg_arrays["all_lat"] if bg_arrays else lat[all_valid]
    else:
        arrays = get_map_arrays(data_source)
        if arrays is None:
            return None
        lon = arrays["lon"]
        lat = arrays["lat"]
        expected = arrays["expected"]
        computed = arrays["computed"]
        x = arrays["x"]
        all_valid = arrays["all_valid"]
        all_lon = arrays["all_lon"]
        all_lat = arrays["all_lat"]
        mask = all_valid

    lon_f = lon[mask]
    lat_f = lat[mask]
    exp_f = expected[mask]
    comp_f = computed[mask]

    # Colour range from filtered points
    all_vals = numpy.concatenate([exp_f, comp_f])
    valid_v = all_vals[~numpy.isnan(all_vals)]
    cmin = float(numpy.nanmin(valid_v)) if len(valid_v) > 0 else 0.0
    cmax = float(numpy.nanmax(valid_v)) if len(valid_v) > 0 else 1.0

    fig = go.Figure()

    # Background: full track in light grey (only when zoomed)
    if has_zoom:
        fig.add_trace(
            go.Scattergeo(
                lon=all_lon,
                lat=all_lat,
                mode="markers",
                marker={"size": 2, "color": "#ccc"},
                name="Full track",
                hoverinfo="skip",
            )
        )

    # Foreground: filtered points in colour
    fig.add_trace(
        go.Scattergeo(
            lon=lon_f,
            lat=lat_f,
            mode="markers",
            marker={
                "size": 4,
                "color": exp_f,
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
    fig.add_trace(
        go.Scattergeo(
            lon=lon_f,
            lat=lat_f,
            mode="markers",
            marker={
                "size": 4,
                "color": comp_f,
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

    # Compute viewport
    view_lon = lon_f[~numpy.isnan(lon_f)] if len(lon_f) > 0 else all_lon
    view_lat = lat_f[~numpy.isnan(lat_f)] if len(lat_f) > 0 else all_lat

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
    if len(view_lon) > 0:
        lon_min = float(view_lon.min())
        lon_max = float(view_lon.max())
        lat_min = float(view_lat.min())
        lat_max = float(view_lat.max())
        lon_span = lon_max - lon_min
        lat_span = lat_max - lat_min
        if lon_span < 3.0:
            pad = (3.0 - lon_span) / 2
            lon_min -= pad
            lon_max += pad
        if lat_span < 3.0:
            pad = (3.0 - lat_span) / 2
            lat_min -= pad
            lat_max += pad
        lon_margin = max(1.0, (lon_max - lon_min) * 0.15)
        lat_margin = max(1.0, (lat_max - lat_min) * 0.15)
        geo_kwargs["lonaxis_range"] = [lon_min - lon_margin, lon_max + lon_margin]
        geo_kwargs["lataxis_range"] = [lat_min - lat_margin, lat_max + lat_margin]

    fig.update_geos(**geo_kwargs)
    fig.update_layout(
        template="seaborn",
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


def build_detail_view(state, varpath, zoom_range=None):
    """Build the detail view for a variable."""
    # Find the long name from comparison results
    long_name = varpath
    var_status = "unknown"
    for gr in state.group_results:
        for v in gr.variables:
            if v.varpath == varpath:
                long_name = v.long_name
                var_status = v.status
                break

    # Load data
    data_source = load_variable_data(state, varpath)

    # Build all Plotly figures
    plot_configs = build_all_figures(long_name, data_source)

    # Previous/Next variable navigation
    clickable_paths = get_clickable_varpaths(state)
    prev_varpath = None
    next_varpath = None
    if varpath in clickable_paths:
        idx = clickable_paths.index(varpath)
        if idx > 0:
            prev_varpath = clickable_paths[idx - 1]
        if idx < len(clickable_paths) - 1:
            next_varpath = clickable_paths[idx + 1]

    # Header
    status_bg = STATUS_COLORS.get(var_status, "#e0e0e0")
    status_fg = STATUS_TEXT_COLORS.get(var_status, "#000")
    status_label = var_status.replace("_", " ").title()

    parent_group = (
        "/" + "/".join(varpath.strip("/").split("/")[:-1])
        if "/" in varpath.strip("/")
        else "/"
    )

    # Navigation buttons row
    nav_buttons: list[Any] = []
    nav_buttons.append(
        html.Button(
            "\u2190 Back to group",
            id="back-to-group-btn",
            n_clicks=0,
            style={
                "border": "1px solid #e0e0e0",
                "borderRadius": "4px",
                "padding": "6px 16px",
                "backgroundColor": "white",
                "cursor": "pointer",
                "fontSize": "12px",
            },
        )
    )
    nav_buttons.append(
        html.Button(
            "\u2190 Previous",
            id="prev-var-btn",
            n_clicks=0,
            disabled=prev_varpath is None,
            style={
                "border": "1px solid #e0e0e0",
                "borderRadius": "4px",
                "padding": "6px 16px",
                "backgroundColor": "white",
                "cursor": "pointer" if prev_varpath else "not-allowed",
                "fontSize": "12px",
                "marginLeft": "8px",
                "opacity": "1" if prev_varpath else "0.4",
            },
        )
    )
    nav_buttons.append(
        html.Button(
            "Next \u2192",
            id="next-var-btn",
            n_clicks=0,
            disabled=next_varpath is None,
            style={
                "border": "1px solid #e0e0e0",
                "borderRadius": "4px",
                "padding": "6px 16px",
                "backgroundColor": "white",
                "cursor": "pointer" if next_varpath else "not-allowed",
                "fontSize": "12px",
                "marginLeft": "4px",
                "opacity": "1" if next_varpath else "0.4",
            },
        )
    )
    # Position indicator
    if varpath in clickable_paths:
        pos = clickable_paths.index(varpath) + 1
        nav_buttons.append(
            html.Span(
                f" {pos}/{len(clickable_paths)}",
                style={
                    "fontSize": "11px",
                    "color": "#90a4ae",
                    "marginLeft": "8px",
                    "alignSelf": "center",
                },
            )
        )

    content: list[Any] = [
        html.Div(
            nav_buttons,
            style={
                "display": "flex",
                "alignItems": "center",
                "marginBottom": "12px",
            },
        ),
        dcc.Store(id="detail-parent-group", data=parent_group),
        dcc.Store(id="prev-varpath", data=prev_varpath),
        dcc.Store(id="next-varpath", data=next_varpath),
        # Variable header
        html.Div(
            [
                html.H3(
                    f"\U0001f4ca {varpath}",
                    style={"margin": "0 0 4px 0", "color": "#263238"},
                ),
                html.Span(long_name, style={"fontSize": "12px", "color": "#607d8b"}),
                html.Span(
                    status_label,
                    style={
                        "display": "inline-block",
                        "padding": "2px 10px",
                        "borderRadius": "12px",
                        "fontSize": "11px",
                        "fontWeight": "600",
                        "backgroundColor": status_bg,
                        "color": status_fg,
                        "marginLeft": "8px",
                    },
                ),
            ],
            style={"marginBottom": "12px"},
        ),
    ]

    # Plot visibility toggles
    plot_labels = [label for label, _ in plot_configs]
    content.append(
        html.Div(
            [
                html.Span(
                    "Visible plots:",
                    style={
                        "fontSize": "12px",
                        "color": "#607d8b",
                        "fontWeight": "600",
                        "marginRight": "8px",
                    },
                ),
                dcc.Checklist(
                    id="plot-visibility",
                    options=[{"label": f" {lbl}", "value": lbl} for lbl in plot_labels],
                    value=["Signal Comparison"],
                    inline=True,
                    style={
                        "fontSize": "12px",
                        "display": "inline-flex",
                        "gap": "12px",
                        "flexWrap": "wrap",
                    },
                    inputStyle={"marginRight": "4px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "padding": "10px 14px",
                "border": "1px solid #e0e0e0",
                "borderRadius": "8px",
                "backgroundColor": "#ffffff",
                "marginBottom": "8px",
                "flexWrap": "wrap",
                "gap": "8px",
            },
        )
    )

    # Flag filter (if applicable)
    if data_source.flags:
        all_options = []
        for flag in data_source.flags:
            all_options.extend(flag.options)
        content.append(
            html.Div(
                [
                    html.Div(
                        "\U0001f3f3 Flag Filters",
                        style={
                            "fontWeight": "bold",
                            "color": "#37474f",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        "Select flag values to keep (unselected = all kept)",
                        style={
                            "fontSize": "10px",
                            "color": "#90a4ae",
                            "marginBottom": "6px",
                            "fontStyle": "italic",
                        },
                    ),
                    dcc.Dropdown(
                        id="flag-dropdown",
                        options=[{"label": o, "value": o} for o in all_options],
                        value=[],
                        multi=True,
                        placeholder="Select flag values to filter...",
                        style={"fontSize": "12px"},
                    ),
                ],
                style={
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "8px",
                    "padding": "12px 16px",
                    "backgroundColor": "#ffffff",
                    "marginBottom": "12px",
                },
            )
        )

    # Outlier detection
    d = data_source.source_data
    diff_arr = numpy.asarray(d["difference"], dtype=float)
    valid_diff = diff_arr[~numpy.isnan(diff_arr)]
    outlier_mean = float(numpy.mean(valid_diff)) if len(valid_diff) > 0 else 0.0
    outlier_std = float(numpy.std(valid_diff)) if len(valid_diff) > 0 else 1.0
    default_n_sigma = 3
    outlier_mask = numpy.abs(diff_arr - outlier_mean) > (default_n_sigma * outlier_std)
    n_outliers = int(numpy.nansum(outlier_mask))

    x_arr = numpy.asarray(d["x"])
    for label, fig in plot_configs:
        if label == "Signal Comparison":
            expected_arr = numpy.asarray(d["expected"], dtype=float)
            computed_arr = numpy.asarray(d["computed"], dtype=float)
            fig.add_trace(
                go.Scattergl(
                    x=x_arr[outlier_mask],
                    y=expected_arr[outlier_mask],
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": "orange",
                        "symbol": "diamond",
                        "line": {"width": 1, "color": "red"},
                    },
                    name="Source (outlier)",
                    hovertemplate="index: %{x}<br>source: %{y}<br><b>OUTLIER</b><extra></extra>",
                    visible=True,
                )
            )
            fig.add_trace(
                go.Scattergl(
                    x=x_arr[outlier_mask],
                    y=computed_arr[outlier_mask],
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": "yellow",
                        "symbol": "diamond",
                        "line": {"width": 1, "color": "red"},
                    },
                    name="Target (outlier)",
                    hovertemplate="index: %{x}<br>target: %{y}<br><b>OUTLIER</b><extra></extra>",
                    visible=True,
                )
            )
        elif label == "Signal Difference":
            fig.add_trace(
                go.Scattergl(
                    x=x_arr[outlier_mask],
                    y=diff_arr[outlier_mask],
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": "orange",
                        "symbol": "diamond",
                        "line": {"width": 1, "color": "red"},
                    },
                    name="Outlier",
                    hovertemplate="index: %{x}<br>difference: %{y}<br><b>OUTLIER</b><extra></extra>",
                    visible=True,
                )
            )

    # Sliders row: marker size + outlier threshold on the same line
    _slider_box_style = {
        "display": "flex",
        "alignItems": "center",
        "gap": "8px",
        "flex": "1",
        "minWidth": "250px",
        "padding": "8px 12px",
        "border": "1px solid #e0e0e0",
        "borderRadius": "8px",
        "backgroundColor": "#ffffff",
    }
    content.append(
        html.Div(
            [
                # Marker size slider
                html.Div(
                    [
                        html.Span(
                            "Marker size:",
                            style={
                                "fontSize": "12px",
                                "color": "#607d8b",
                                "fontWeight": "600",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Div(
                            dcc.Slider(
                                id="marker-size-slider",
                                min=1,
                                max=10,
                                step=1,
                                value=2,
                                marks={i: str(i) for i in range(1, 11)},
                            ),
                            style={"flex": "1", "minWidth": "180px"},
                        ),
                    ],
                    style=_slider_box_style,
                ),
                # Outlier threshold slider
                html.Div(
                    [
                        html.Span(
                            "Outlier threshold (N\u00d7\u03c3):",
                            style={
                                "fontSize": "12px",
                                "color": "#607d8b",
                                "fontWeight": "600",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Div(
                            dcc.Slider(
                                id="outlier-sigma-slider",
                                min=1,
                                max=10,
                                step=0.5,
                                value=default_n_sigma,
                                marks={i: str(i) for i in range(1, 11)},
                            ),
                            style={"flex": "1", "minWidth": "180px"},
                        ),
                        # Outlier count badge
                        html.Span(
                            f"Outliers: {n_outliers}",
                            id="outlier-count-display",
                            style={
                                "fontSize": "11px",
                                "color": "#c62828" if n_outliers > 0 else "#2e7d32",
                                "fontWeight": "600",
                                "padding": "3px 10px",
                                "borderRadius": "12px",
                                "backgroundColor": "#ffcdd2"
                                if n_outliers > 0
                                else "#c8e6c9",
                                "whiteSpace": "nowrap",
                            },
                        ),
                    ],
                    style=_slider_box_style,
                ),
                dcc.Store(
                    id="outlier-stats", data={"mean": outlier_mean, "std": outlier_std}
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "stretch",
                "flexWrap": "wrap",
                "gap": "10px",
                "marginBottom": "12px",
            },
        )
    )

    # Apply saved zoom range to all figures
    if zoom_range and not zoom_range.get("auto"):
        xmin = zoom_range.get("min")
        xmax = zoom_range.get("max")
        if xmin is not None and xmax is not None:
            for _label, fig in plot_configs:
                layout = fig.to_plotly_json().get("layout", {})
                xaxis = layout.get("xaxis", {})
                title = xaxis.get("title", "")
                if isinstance(title, dict):
                    title = title.get("text", "")
                if title == "Index":
                    fig.update_layout(xaxis_range=[xmin, xmax])
            filtered_map = build_filtered_map(data_source, zoom_range)
            if filtered_map is not None:
                plot_configs = [
                    (label, filtered_map)
                    if label == "Geographical Position"
                    else (label, fig)
                    for label, fig in plot_configs
                ]

    # Plot graphs
    _default_visible = {"Signal Comparison"}
    for label, fig in plot_configs:
        visible = label in _default_visible
        content.append(
            html.Div(
                [
                    html.H4(
                        label,
                        style={
                            "color": "#455a64",
                            "margin": "0 0 4px 0",
                            "borderBottom": "1px solid #e0e0e0",
                            "paddingBottom": "4px",
                        },
                    ),
                    dcc.Graph(
                        id={"type": "plot-graph", "label": label},
                        figure=fig,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                        },
                        style={"width": "100%"},
                    ),
                ],
                id={"type": "plot-container", "label": label},
                style={"marginBottom": "16px"} if visible else {"display": "none"},
            )
        )

    return content


def render_current_view(state, zoom_range=None):
    """Render the current view (group or detail) based on state.

    Returns:
        Tuple of (breadcrumb_items, main_content_children).
    """
    if state.current_view == "detail" and state.current_varpath:
        varpath = state.current_varpath
        parent = (
            "/" + "/".join(varpath.strip("/").split("/")[:-1])
            if "/" in varpath.strip("/")
            else "/"
        )
        breadcrumb = build_breadcrumb(parent)
        detail = build_detail_view(state, varpath, zoom_range=zoom_range)
        return breadcrumb, detail
    else:
        group_path = state.current_group_path or "/"
        breadcrumb = build_breadcrumb(group_path)
        browser = build_group_browser(state, group_path)
        return breadcrumb, browser
