"""Dash callbacks for the NetCDF comparison application."""

from __future__ import annotations

from typing import TYPE_CHECKING
import json
import logging

from dash import (
    ALL,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    html,
    no_update,
)
import numpy
import plotly.graph_objects as go
import plotly.io as pio

if TYPE_CHECKING:
    from .state import SessionState
    import dash

from .comparison import FilterInfo, scan_all_groups
from .constants import STATUS_COLORS, STATUS_TEXT_COLORS
from .data_loading import load_variable_data
from .expression import (
    create_expression_mask,
    find_affected_groups,
    rewrite_expression,
)
from .plots import build_all_figures

LOGGER = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _find_group_result(state, group_path):
    for gr in state.group_results:
        if gr.group_path == group_path:
            return gr
    return None


def _navigate_to_group(dataset, group_path):
    if group_path == "/":
        return dataset
    parts = group_path.strip("/").split("/")
    group = dataset
    for part in parts:
        if part in group.groups:
            group = group.groups[part]
        else:
            return None
    return group


def _count_diffs(state, group_path):
    gr = _find_group_result(state, group_path)
    if gr is None:
        return 0, 0
    n_diff = sum(1 for v in gr.variables
                 if v.status in ("different", "almost_equal"))
    return n_diff, len(gr.variables)


def _count_diffs_recursive(state, dataset, group_path):
    n_diff, n_total = _count_diffs(state, group_path)
    group = _navigate_to_group(dataset, group_path)
    if group is not None:
        for grp_name in group.groups:
            child_path = (f"{group_path}/{grp_name}"
                          if group_path != "/" else f"/{grp_name}")
            cd, ct = _count_diffs_recursive(state, dataset, child_path)
            n_diff += cd
            n_total += ct
    return n_diff, n_total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Filter helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _conditions_to_expression(conditions: list[dict]) -> str:
    """Convert a list of {var, op, val} dicts to an expression string."""
    parts = []
    for c in conditions:
        var_ref = f"${{{c['var']}}}"
        if c["op"] == "in":
            parts.append(f"np.isin({var_ref}, {c['val']})")
        else:
            parts.append(f"({var_ref} {c['op']} {c['val']})")
    return " & ".join(parts)


_STATUS_OK = {
    "fontSize": "12px", "padding": "8px 12px",
    "borderRadius": "4px", "backgroundColor": "#c8e6c9",
    "color": "#2e7d32",
}
_STATUS_ERR = {
    "fontSize": "12px", "padding": "8px 12px",
    "borderRadius": "4px", "backgroundColor": "#ffcdd2",
    "color": "#c62828",
}
_STATUS_NONE = {
    "fontSize": "12px", "padding": "8px 12px",
    "borderRadius": "4px", "backgroundColor": "#e3f2fd",
    "color": "#1565c0",
}

_BANNER_ACTIVE = {
    "backgroundColor": "#ff9800", "color": "white",
    "padding": "4px 14px", "borderRadius": "14px",
    "fontSize": "12px", "fontWeight": "600",
    "letterSpacing": "0.3px",
    "display": "inline-flex", "alignItems": "center", "gap": "6px",
}
_BANNER_HIDDEN = {"display": "none"}


def _make_banner(state: SessionState):
    """Return (children, style) for the filter-banner element."""
    if state.active_mask is not None:
        selected = int(state.active_mask.sum())
        total = int(state.active_mask.size)
        masked = total - selected
        pct = 100 * masked / total if total > 0 else 0
        text = f"\u26a0 Filter active \u2014 {masked:,}/{total:,} masked ({pct:.1f}%)"
        return text, _BANNER_ACTIVE
    return "", _BANNER_HIDDEN



def _apply_expression_no_nav(state: SessionState, raw: str):
    """Apply an expression filter. Returns (status_msg, status_style) or None.

    Updates state in-place (mask, cache, group_results).
    """
    LOGGER.info("_apply_expression_no_nav: raw=%r", raw)
    rewritten = rewrite_expression(raw)
    LOGGER.info("_apply_expression_no_nav: rewritten=%r", rewritten)
    mask, error = create_expression_mask(state.ref_ds, rewritten)
    if mask is None:
        LOGGER.error("_apply_expression_no_nav: error=%s", error)
        return (f"Error: {error}", _STATUS_ERR)

    affected = find_affected_groups(raw)
    state.active_mask = mask
    state.active_expression = raw
    state.clear_variable_cache()
    selected = int(mask.sum())
    total = int(mask.size)
    masked = total - selected
    LOGGER.info("_apply_expression_no_nav: %d/%d selected, mask id=%d",
                selected, total, id(mask))
    state.filter_info = FilterInfo(
        expression=raw,
        selected_points=selected,
        total_points=total,
        affected_groups=affected,
    )
    state.group_results = scan_all_groups(
        state.ref_ds, state.tgt_ds, state.filter_info,
        mask=state.active_mask)
    msg = (f"Filter active: {masked:,}/{total:,} points masked "
           f"({100 * masked / total:.1f}%)")
    return (msg, _STATUS_OK)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  View builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_breadcrumb(group_path):
    """Build clickable breadcrumb components."""
    parts = [p for p in group_path.split("/") if p]
    items = []

    items.append(html.Button(
        "\U0001f3e0", id={"type": "breadcrumb-btn", "path": "/"},
        n_clicks=0,
        style={
            "border": "none", "background": "transparent",
            "cursor": "pointer", "fontSize": "16px", "padding": "4px 8px",
        },
    ))

    accumulated = ""
    for i, part in enumerate(parts):
        accumulated += f"/{part}"
        items.append(html.Span(
            "\u203a", style={
                "color": "#90a4ae", "margin": "0 4px", "fontSize": "16px",
            }))
        is_last = (i == len(parts) - 1)
        items.append(html.Button(
            part,
            id={"type": "breadcrumb-btn", "path": accumulated},
            n_clicks=0,
            style={
                "border": "none", "background": "transparent",
                "cursor": "pointer", "fontSize": "13px",
                "padding": "4px 8px",
                "fontWeight": "600" if is_last else "400",
                "color": "#263238" if is_last else "#607d8b",
            },
        ))

    return html.Div(items, style={
        "background": "linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%)",
        "borderRadius": "6px", "padding": "6px 12px",
        "marginBottom": "12px", "display": "flex", "alignItems": "center",
        "flexWrap": "wrap",
    })


def _get_clickable_varpaths(state):
    """Build a flat list of all clickable (different/almost_equal) varpaths."""
    paths = []
    for gr in state.group_results:
        paths.extend(v.varpath for v in gr.variables
                        if v.status in ("different", "almost_equal"))
    return paths


def _build_global_summary(state):
    """Build a summary table of ALL problematic variables across all groups."""
    problem_vars = []
    for gr in state.group_results:
        problem_vars.extend(v for v in gr.variables
                            if v.status in ("different", "almost_equal") and v.stats is not None)
    if not problem_vars:
        return []

    # Sort by max absolute error descending
    problem_vars.sort(
        key=lambda v: abs(float(v.stats.max_diff)), reverse=True)

    content = []
    content.append(html.H4(
        "\u26a0 Global Summary \u2014 Variables with Differences",
        style={"color": "#c62828", "margin": "16px 0 8px 0"}))
    content.append(html.Div([
        html.Span(
            f"{len(problem_vars)} variable(s) with differences across all "
            "groups. Sorted by max |diff| descending. Click a row to inspect.",
            style={"fontSize": "11px", "color": "#78909c"}),
        dcc.Input(
            id="summary-search-input",
            type="text",
            placeholder="Filter variables\u2026",
            debounce=True,
            style={
                "marginLeft": "16px", "padding": "4px 10px",
                "border": "1px solid #cfd8dc", "borderRadius": "4px",
                "fontSize": "12px", "width": "250px",
            },
        ),
    ], style={"display": "flex", "alignItems": "center",
              "marginBottom": "6px"}))

    rows = []
    for v in problem_vars:
        bg = STATUS_COLORS.get(v.status, "#ffffff")
        fg = STATUS_TEXT_COLORS.get(v.status, "#000000")
        cell_style = {"padding": "6px 12px", "fontSize": "12px"}
        rows.append(html.Tr([
            html.Td(v.varpath, style=cell_style),
            html.Td(v.status.replace("_", " ").title(), style=cell_style),
            html.Td(f"{abs(float(v.stats.max_diff)):.6g}", style=cell_style),
            html.Td(f"{abs(float(v.stats.mean_diff)):.6g}", style=cell_style),
            html.Td(f"{float(v.stats.std_diff):.6g}", style=cell_style),
        ], id={"type": "summary-var-row", "path": v.varpath},
            n_clicks=0,
            style={
                "backgroundColor": bg, "color": fg,
                "cursor": "pointer",
            },
        ))

    content.append(html.Table([
        html.Thead(html.Tr([
            html.Th("Variable Path", style={"padding": "8px 12px",
                     "textAlign": "left"}),
            html.Th("Status", style={"padding": "8px 12px",
                     "textAlign": "left"}),
            html.Th("Max |diff|", style={"padding": "8px 12px",
                     "textAlign": "left"}),
            html.Th("Mean |diff|", style={"padding": "8px 12px",
                     "textAlign": "left"}),
            html.Th("Std", style={"padding": "8px 12px",
                     "textAlign": "left"}),
        ], style={"backgroundColor": "#1565c0", "color": "white"})),
        html.Tbody(rows, id="summary-table-body"),
    ], style={
        "width": "100%", "borderCollapse": "collapse",
        "border": "1px solid #e0e0e0", "fontSize": "12px",
        "marginBottom": "20px",
    }))

    return content


def _build_group_browser(state, group_path):
    """Build group browser view content."""
    gr = _find_group_result(state, group_path)
    content = []

    # At root: show a toggle between Summary and Browse views
    if group_path == "/":
        summary_items = _build_global_summary(state)
        if summary_items:
            content.append(html.Div([
                html.Button(
                    "\u26a0 Summary",
                    id="root-tab-summary",
                    n_clicks=0,
                    style={
                        "border": "none", "borderBottom": "3px solid #1565c0",
                        "padding": "8px 20px", "cursor": "pointer",
                        "fontSize": "13px", "fontWeight": "600",
                        "backgroundColor": "transparent", "color": "#1565c0",
                    },
                ),
                html.Button(
                    "\U0001f4c2 Browse groups",
                    id="root-tab-browse",
                    n_clicks=0,
                    style={
                        "border": "none", "borderBottom": "3px solid transparent",
                        "padding": "8px 20px", "cursor": "pointer",
                        "fontSize": "13px", "fontWeight": "400",
                        "backgroundColor": "transparent", "color": "#78909c",
                    },
                ),
            ], style={
                "display": "flex", "gap": "4px",
                "borderBottom": "1px solid #e0e0e0",
                "marginBottom": "12px",
            }))
            # Summary panel (visible by default)
            content.append(html.Div(
                summary_items,
                id="root-panel-summary",
                style={"display": "block"},
            ))
    if gr is None:
        content.append(html.Div(
            f"Group '{group_path}' not found in comparison results.",
            style={
                "padding": "12px", "backgroundColor": "#fff3cd",
                "borderRadius": "4px", "color": "#856404",
            }))
        return content

    # Summary bar
    n_diff = sum(1 for v in gr.variables
                 if v.status in ("different", "almost_equal"))
    n_eq = sum(1 for v in gr.variables if v.status == "equal")
    n_aeq = sum(1 for v in gr.variables if v.status == "almost_equal")
    n_masked = sum(1 for v in gr.variables if v.status == "masked")

    summary_parts = []
    if n_eq:
        summary_parts.append(
            html.Span(f"\u2713 {n_eq} equal",
                      style={"color": "#2e7d32"}))
    if n_aeq:
        summary_parts.append(
            html.Span(f"\u2248 {n_aeq} almost equal",
                      style={"color": "#006400"}))
    if n_diff:
        summary_parts.append(
            html.Span(f"\u2260 {n_diff} different",
                      style={"color": "#e65100"}))
    if n_masked:
        summary_parts.append(
            html.Span(f"\u2014 {n_masked} masked",
                      style={"color": "#757575"}))

    if summary_parts:
        separator = html.Span(" \u2022 ", style={"color": "#bbb"})
        parts_with_sep = []
        for i, sp in enumerate(summary_parts):
            if i > 0:
                parts_with_sep.append(separator)
            parts_with_sep.append(sp)

        content.append(html.Div([
            html.B(f"{len(gr.variables)}"), " variables: ", *parts_with_sep,
        ], style={
            "fontSize": "13px", "padding": "8px 12px",
            "background": "#f5f7fa", "borderRadius": "6px",
            "marginBottom": "12px",
        }))

    # Filter info
    if gr.filter_description:
        content.append(html.Div(
            f"\U0001f50d {gr.filter_description}",
            style={
                "padding": "8px 12px", "backgroundColor": "#e3f2fd",
                "borderRadius": "4px", "color": "#1565c0",
                "fontSize": "12px", "marginBottom": "12px",
            }))

    # Sub-groups as buttons
    ref_group = _navigate_to_group(state.ref_ds, group_path)
    if ref_group is not None and ref_group.groups:
        content.append(html.H4(
            "\U0001f4c2 Sub-groups",
            style={"color": "#37474f", "margin": "16px 0 8px 0"}))
        btn_row = []
        for grp_name in sorted(ref_group.groups):
            child_path = (f"{group_path}/{grp_name}"
                          if group_path != "/" else f"/{grp_name}")
            n_d, _n_t = _count_diffs_recursive(
                state, state.ref_ds, child_path)
            label = f"\U0001f4c1 {grp_name}"
            if n_d > 0:
                label += f" ({n_d} diffs)"
            bg_color = "#fff3cd" if n_d > 0 else "#f5f7fa"
            border_color = "#ffc107" if n_d > 0 else "#e0e0e0"
            btn_row.append(html.Button(
                label,
                id={"type": "group-btn", "path": child_path},
                n_clicks=0,
                style={
                    "border": f"1px solid {border_color}",
                    "borderRadius": "4px", "padding": "6px 12px",
                    "backgroundColor": bg_color,
                    "cursor": "pointer", "fontSize": "12px",
                    "marginRight": "6px", "marginBottom": "6px",
                    "textAlign": "left",
                },
            ))
        content.append(html.Div(
            btn_row, style={"display": "flex", "flexWrap": "wrap"}))

    # Attributes table
    if gr.attributes:
        content.append(html.H4(
            "\U0001f4cb Attributes",
            style={"color": "#37474f", "margin": "16px 0 8px 0"}))
        rows = []
        rows.extend([
            html.Tr([
                html.Td(a.name, style={"padding": "6px 12px"}),
                html.Td(str(a.ref_value), style={"padding": "6px 12px"}),
                html.Td(str(a.tgt_value), style={"padding": "6px 12px"}),
            ]) for a in gr.attributes
        ])
        content.append(html.Table([
            html.Thead(html.Tr([
                html.Th("Attribute", style={"padding": "8px 12px",
                         "textAlign": "left"}),
                html.Th("Reference Value", style={"padding": "8px 12px",
                         "textAlign": "left"}),
                html.Th("Target Value", style={"padding": "8px 12px",
                         "textAlign": "left"}),
            ], style={"backgroundColor": "#f5f7fa"})),
            html.Tbody(rows),
        ], style={
            "width": "100%", "borderCollapse": "collapse",
            "border": "1px solid #e0e0e0", "marginBottom": "12px",
            "fontSize": "12px",
        }))

    # Variables table
    if gr.variables:
        content.append(html.H4(
            "\U0001f4ca Variables",
            style={"color": "#37474f", "margin": "16px 0 8px 0"}))
        if n_diff > 0:
            content.append(html.Div(
                "Click a coloured row to see detailed comparison plots",
                style={
                    "fontSize": "11px", "color": "#78909c",
                    "marginBottom": "6px",
                }))

        # Feature 2: Search field for filtering variables
        content.append(html.Div([
            dcc.Input(
                id="var-search-input",
                type="text",
                placeholder="Search variables...",
                debounce=False,
                style={
                    "width": "300px", "padding": "6px 12px",
                    "border": "1px solid #e0e0e0", "borderRadius": "4px",
                    "fontSize": "12px", "marginBottom": "8px",
                },
            ),
        ]))

        var_rows = []
        for v in gr.variables:
            value_info = v.message or v.status.replace("_", " ").title()
            if v.stats is not None:
                value_info = f"{v.stats}, atol={v.atol}"
            short_name = v.varpath.rsplit("/", 1)[-1]
            bg = STATUS_COLORS.get(v.status, "#ffffff")
            fg = STATUS_TEXT_COLORS.get(v.status, "#000000")
            clickable = v.status in ("different", "almost_equal")
            row_style = {
                "backgroundColor": bg, "color": fg,
                "cursor": "pointer" if clickable else "default",
            }
            cell_style = {"padding": "6px 12px", "fontSize": "12px"}
            var_rows.append(html.Tr([
                html.Td(short_name, style=cell_style),
                html.Td(v.status.replace("_", " ").title(),
                         style=cell_style),
                html.Td("\u2260" if v.shape_diff else "\u2713",
                         style=cell_style),
                html.Td("\u2260" if v.dtype_diff else "\u2713",
                         style=cell_style),
                html.Td(value_info, style={
                    **cell_style, "maxWidth": "300px",
                    "overflow": "hidden", "textOverflow": "ellipsis",
                }),
            ], id={"type": "var-row", "path": v.varpath},
                n_clicks=0,
                style=row_style,
            ))

        content.append(html.Div(
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Variable", style={"padding": "8px 12px",
                             "textAlign": "left"}),
                    html.Th("Status", style={"padding": "8px 12px",
                             "textAlign": "left"}),
                    html.Th("Shape", style={"padding": "8px 12px",
                             "textAlign": "left"}),
                    html.Th("DType", style={"padding": "8px 12px",
                             "textAlign": "left"}),
                    html.Th("Values", style={"padding": "8px 12px",
                             "textAlign": "left"}),
                ], style={"backgroundColor": "#1565c0", "color": "white"})),
                html.Tbody(var_rows, id="var-table-body"),
            ], style={
                "width": "100%", "borderCollapse": "collapse",
                "border": "1px solid #e0e0e0", "fontSize": "12px",
            }),
            id="var-table-container",
        ))

    # At root with summary: wrap group browser items into the Browse panel
    if group_path == "/" and any(
            getattr(c, "id", None) == "root-panel-summary"
            for c in content):
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
        tabs_and_summary.append(html.Div(
            browse_items,
            id="root-panel-browse",
            style={"display": "none"},
        ))
        return tabs_and_summary

    return content


def _build_filtered_map(data_source, zoom_range):
    """Build a Scattergeo figure filtered by zoom_range.

    All points are shown in light grey for context.  Points within the
    zoomed index range are overlaid in colour.  The geographic viewport
    zooms to the filtered points but is forced to a square aspect ratio
    so the rendered map rectangle keeps a consistent size.
    """
    d = data_source.source_data
    if "longitude" not in d:
        return None

    lon = numpy.asarray(d["longitude"], dtype=float)
    lat = numpy.asarray(d["latitude"], dtype=float)
    expected = numpy.asarray(d["expected"], dtype=float)
    computed = numpy.asarray(d["computed"], dtype=float)
    x = numpy.asarray(d["x"], dtype=float)

    all_valid = ~(numpy.isnan(lon) | numpy.isnan(lat))
    all_lon = lon[all_valid]
    all_lat = lat[all_valid]

    has_zoom = (zoom_range and not zoom_range.get("auto"))

    if has_zoom:
        xmin = float(zoom_range["min"])
        xmax = float(zoom_range["max"])
        mask = ((x >= xmin) & (x <= xmax) & all_valid)
    else:
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
        fig.add_trace(go.Scattergeo(
            lon=all_lon, lat=all_lat, mode="markers",
            marker={"size": 2, "color": "#ccc"},
            name="Full track",
            hoverinfo="skip",
        ))

    # Foreground: filtered points in colour
    fig.add_trace(go.Scattergeo(
        lon=lon_f, lat=lat_f, mode="markers",
        marker={"size": 4, "color": exp_f, "colorscale": "Viridis",
                "cmin": cmin, "cmax": cmax, "showscale": False},
        name="Reference",
        hovertemplate=("lon: %{lon:.3f}<br>lat: %{lat:.3f}"
                       "<br>reference: %{marker.color:.4g}"
                       "<extra></extra>"),
    ))
    fig.add_trace(go.Scattergeo(
        lon=lon_f, lat=lat_f, mode="markers",
        marker={"size": 4, "color": comp_f, "colorscale": "Viridis",
                "cmin": cmin, "cmax": cmax, "showscale": False},
        name="Target",
        visible="legendonly",
        hovertemplate=("lon: %{lon:.3f}<br>lat: %{lat:.3f}"
                       "<br>target: %{marker.color:.4g}"
                       "<extra></extra>"),
    ))

    # Compute viewport: zoom to filtered points, force square aspect
    view_lon = lon_f[~numpy.isnan(lon_f)] if len(lon_f) > 0 else all_lon
    view_lat = lat_f[~numpy.isnan(lat_f)] if len(lat_f) > 0 else all_lat

    geo_kwargs = {
        "showcoastlines": True, "coastlinecolor": "#555",
        "showland": True, "landcolor": "#e8e8e8",
        "showocean": True, "oceancolor": "#d4ecf7",
        "showlakes": True, "lakecolor": "#d4ecf7",
        "showcountries": True, "countrycolor": "#aaa",
        "projection_type": "equirectangular",
    }
    if len(view_lon) > 0:
        lon_mid = float((view_lon.min() + view_lon.max()) / 2)
        lat_mid = float((view_lat.min() + view_lat.max()) / 2)
        lon_span = float(view_lon.max() - view_lon.min())
        lat_span = float(view_lat.max() - view_lat.min())
        # Force square: take the largest span, enforce a minimum of 5°
        half = max(lon_span, lat_span, 5.0) / 2 + 2.0
        geo_kwargs["lonaxis_range"] = [lon_mid - half, lon_mid + half]
        geo_kwargs["lataxis_range"] = [lat_mid - half, lat_mid + half]
        geo_kwargs["projection_rotation_lon"] = lon_mid
        geo_kwargs["projection_rotation_lat"] = lat_mid

    fig.update_geos(**geo_kwargs)
    fig.update_layout(
        height=500,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0, "itemsizing": "constant"},
    )
    return fig


def _build_detail_view(state, varpath, zoom_range=None):
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

    # Feature 1: Previous/Next variable navigation
    clickable_paths = _get_clickable_varpaths(state)
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

    parent_group = "/" + "/".join(
        varpath.strip("/").split("/")[:-1]) if "/" in varpath.strip("/") else "/"

    # Navigation buttons row
    nav_buttons = []
    nav_buttons.append(html.Button(
        "\u2190 Back to group",
        id="back-to-group-btn",
        n_clicks=0,
        style={
            "border": "1px solid #e0e0e0", "borderRadius": "4px",
            "padding": "6px 16px", "backgroundColor": "white",
            "cursor": "pointer", "fontSize": "12px",
        },
    ))
    nav_buttons.append(html.Button(
        "\u2190 Previous",
        id="prev-var-btn",
        n_clicks=0,
        disabled=prev_varpath is None,
        style={
            "border": "1px solid #e0e0e0", "borderRadius": "4px",
            "padding": "6px 16px", "backgroundColor": "white",
            "cursor": "pointer" if prev_varpath else "not-allowed",
            "fontSize": "12px", "marginLeft": "8px",
            "opacity": "1" if prev_varpath else "0.4",
        },
    ))
    nav_buttons.append(html.Button(
        "Next \u2192",
        id="next-var-btn",
        n_clicks=0,
        disabled=next_varpath is None,
        style={
            "border": "1px solid #e0e0e0", "borderRadius": "4px",
            "padding": "6px 16px", "backgroundColor": "white",
            "cursor": "pointer" if next_varpath else "not-allowed",
            "fontSize": "12px", "marginLeft": "4px",
            "opacity": "1" if next_varpath else "0.4",
        },
    ))
    # Position indicator
    if varpath in clickable_paths:
        pos = clickable_paths.index(varpath) + 1
        nav_buttons.append(html.Span(
            f" {pos}/{len(clickable_paths)}",
            style={"fontSize": "11px", "color": "#90a4ae",
                   "marginLeft": "8px", "alignSelf": "center"},
        ))

    content = [
        html.Div(nav_buttons, style={
            "display": "flex", "alignItems": "center",
            "marginBottom": "12px",
        }),
        dcc.Store(id="detail-parent-group", data=parent_group),
        dcc.Store(id="prev-varpath", data=prev_varpath),
        dcc.Store(id="next-varpath", data=next_varpath),
        # Variable header
        html.Div([
            html.H3(f"\U0001f4ca {varpath}", style={
                "margin": "0 0 4px 0", "color": "#263238"}),
            html.Span(long_name, style={
                "fontSize": "12px", "color": "#607d8b"}),
            html.Span(status_label, style={
                "display": "inline-block", "padding": "2px 10px",
                "borderRadius": "12px", "fontSize": "11px",
                "fontWeight": "600", "backgroundColor": status_bg,
                "color": status_fg, "marginLeft": "8px",
            }),
        ], style={"marginBottom": "12px"}),
    ]

    # Plot visibility toggles + marker size control
    plot_labels = [label for label, _ in plot_configs]
    content.append(html.Div([
        html.Div([
            html.Span("Visible plots:", style={
                "fontSize": "12px", "color": "#607d8b",
                "fontWeight": "600", "marginRight": "8px",
            }),
            dcc.Checklist(
                id="plot-visibility",
                options=[{"label": f" {lbl}", "value": lbl}
                         for lbl in plot_labels],
                value=["Signal Comparison"],
                inline=True,
                style={"fontSize": "12px", "display": "inline-flex",
                       "gap": "12px", "flexWrap": "wrap"},
                inputStyle={"marginRight": "4px"},
            ),
        ], style={"flex": "1"}),
        html.Div([
            html.Span("Marker size:", style={
                "fontSize": "12px", "color": "#607d8b",
                "fontWeight": "600", "marginRight": "8px",
            }),
            dcc.Slider(
                id="marker-size-slider",
                min=1, max=10, step=1, value=2,
                marks={i: str(i) for i in range(1, 11)},
            ),
        ], style={"width": "300px"}),
    ], style={
        "display": "flex", "alignItems": "center",
        "padding": "10px 14px",
        "border": "1px solid #e0e0e0", "borderRadius": "8px",
        "backgroundColor": "#ffffff", "marginBottom": "12px",
        "flexWrap": "wrap", "gap": "12px",
    }))

    # Flag filter (if applicable)
    if data_source.flags:
        all_options = []
        for flag in data_source.flags:
            all_options.extend(flag.options)
        content.append(html.Div([
            html.Div("\U0001f3f3 Flag Filters", style={
                "fontWeight": "bold", "color": "#37474f",
                "marginBottom": "4px",
            }),
            html.Div(
                "Select flag values to keep (unselected = all kept)",
                style={"fontSize": "10px", "color": "#90a4ae",
                       "marginBottom": "6px", "fontStyle": "italic"}),
            dcc.Dropdown(
                id="flag-dropdown",
                options=[{"label": o, "value": o} for o in all_options],
                value=[],
                multi=True,
                placeholder="Select flag values to filter...",
                style={"fontSize": "12px"},
            ),
        ], style={
            "border": "1px solid #e0e0e0", "borderRadius": "8px",
            "padding": "12px 16px", "backgroundColor": "#ffffff",
            "marginBottom": "12px",
        }))

    # Feature 4: Outlier detection - add outlier traces to signal plots
    d = data_source.source_data
    diff_arr = numpy.asarray(d["difference"], dtype=float)
    valid_diff = diff_arr[~numpy.isnan(diff_arr)]
    outlier_mean = float(numpy.mean(valid_diff)) if len(valid_diff) > 0 else 0.0
    outlier_std = float(numpy.std(valid_diff)) if len(valid_diff) > 0 else 1.0
    default_n_sigma = 3
    outlier_mask = numpy.abs(diff_arr - outlier_mean) > (
        default_n_sigma * outlier_std)
    n_outliers = int(numpy.nansum(outlier_mask))

    # Add outlier traces to signal comparison and signal difference
    x_arr = numpy.asarray(d["x"])
    for label, fig in plot_configs:
        if label == "Signal Comparison":
            expected_arr = numpy.asarray(d["expected"], dtype=float)
            computed_arr = numpy.asarray(d["computed"], dtype=float)
            fig.add_trace(go.Scattergl(
                x=x_arr[outlier_mask],
                y=expected_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "orange",
                            "symbol": "diamond", "line": {"width": 1, "color": "red"}},
                name="Reference (outlier)",
                hovertemplate=(
                    "index: %{x}<br>reference: %{y}<br>"
                    "<b>OUTLIER</b><extra></extra>"),
                visible=True,
            ))
            fig.add_trace(go.Scattergl(
                x=x_arr[outlier_mask],
                y=computed_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "yellow",
                            "symbol": "diamond", "line": {"width": 1, "color": "red"}},
                name="Target (outlier)",
                hovertemplate=(
                    "index: %{x}<br>target: %{y}<br>"
                    "<b>OUTLIER</b><extra></extra>"),
                visible=True,
            ))
        elif label == "Signal Difference":
            fig.add_trace(go.Scattergl(
                x=x_arr[outlier_mask],
                y=diff_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "orange",
                            "symbol": "diamond", "line": {"width": 1, "color": "red"}},
                name="Outlier",
                hovertemplate=(
                    "index: %{x}<br>difference: %{y}<br>"
                    "<b>OUTLIER</b><extra></extra>"),
                visible=True,
            ))

    # Outlier controls
    content.append(html.Div([
        html.Div([
            html.Span("Outlier threshold (N\u00d7\u03c3):", style={
                "fontSize": "12px", "color": "#607d8b",
                "fontWeight": "600", "marginRight": "8px",
            }),
            dcc.Slider(
                id="outlier-sigma-slider",
                min=1, max=10, step=0.5, value=default_n_sigma,
                marks={i: str(i) for i in range(1, 11)},
            ),
        ], style={"width": "400px"}),
        html.Div([
            html.Span(
                f"Outliers detected: {n_outliers}",
                id="outlier-count-display",
                style={
                    "fontSize": "12px", "color": "#c62828" if n_outliers > 0
                    else "#2e7d32",
                    "fontWeight": "600", "padding": "4px 12px",
                    "borderRadius": "12px",
                    "backgroundColor": "#ffcdd2" if n_outliers > 0
                    else "#c8e6c9",
                }),
        ], style={"marginLeft": "20px", "display": "flex",
                  "alignItems": "center"}),
        # Store the outlier statistics for clientside callback
        dcc.Store(id="outlier-stats", data={
            "mean": outlier_mean, "std": outlier_std,
        }),
        html.Div(
            "Outliers: points where |difference − mean| > N×σ. "
            "On the Signal Comparison plot, orange diamonds mark the "
            "Reference value and yellow diamonds the Target value "
            "at detected positions.",
            style={
                "fontSize": "11px", "color": "#90a4ae",
                "fontStyle": "italic",  "marginTop": "6px",
                "lineHeight": "1.4",
            },
        ),
    ], style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "padding": "10px 14px",
        "border": "1px solid #e0e0e0", "borderRadius": "8px",
        "backgroundColor": "#ffffff", "marginBottom": "12px",
    }))

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
            # Rebuild geographical map filtered to zoom range
            filtered_map = _build_filtered_map(data_source, zoom_range)
            if filtered_map is not None:
                plot_configs = [
                    (label, filtered_map)
                    if label == "Geographical Position"
                    else (label, fig)
                    for label, fig in plot_configs
                ]

    # Plot graphs - each wrapped in a togglable div, with stable graph ids
    _default_visible = {"Signal Comparison"}
    for label, fig in plot_configs:
        visible = label in _default_visible
        content.append(html.Div([
            html.H4(label, style={
                "color": "#455a64", "margin": "0 0 4px 0",
                "borderBottom": "1px solid #e0e0e0",
                "paddingBottom": "4px",
            }),
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
        ], id={"type": "plot-container", "label": label},
            style={"marginBottom": "16px"} if visible
            else {"display": "none"}))

    return content


def _render_current_view(state, zoom_range=None):
    """Render the current view (group or detail) based on state."""
    if state.current_view == "detail" and state.current_varpath:
        varpath = state.current_varpath
        parent = ("/" + "/".join(varpath.strip("/").split("/")[:-1])
                  if "/" in varpath.strip("/") else "/")
        breadcrumb = _build_breadcrumb(parent)
        detail = _build_detail_view(state, varpath, zoom_range=zoom_range)
        return [breadcrumb, *detail]
    else:
        group_path = state.current_group_path or "/"
        breadcrumb = _build_breadcrumb(group_path)
        browser = _build_group_browser(state, group_path)
        return [breadcrumb, *browser]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Callback registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def register_callbacks(app: dash.Dash, state: SessionState) -> None:
    """Register all Dash callbacks.

    Args:
        app: The Dash application.
        state: The shared session state.

    """

    # ── Main content rendering ─────────────────────────────────────

    @app.callback(
        Output("main-content", "children"),
        Input("nav-current-view", "data"),
        Input("nav-group-path", "data"),
        Input("nav-varpath", "data"),
        State("zoom-range", "data"),
    )
    def render_main_content(view, group_path, varpath, zoom_range):
        if view == "detail" and varpath:
            state.current_view = "detail"
            state.current_varpath = varpath
        else:
            state.current_view = "group"
            state.current_group_path = group_path or "/"
        return _render_current_view(state, zoom_range=zoom_range)

    # ── Navigation: group buttons ──────────────────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Input({"type": "group-btn", "path": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_group_click(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        triggered = ctx.triggered[0]
        if triggered["value"] is None or triggered["value"] == 0:
            return no_update, no_update, no_update
        prop_id = triggered["prop_id"]
        # Extract path from the pattern-matching ID
        id_str = prop_id.rsplit(".", 1)[0]
        id_dict = json.loads(id_str)
        path = id_dict["path"]
        return "group", path, None

    # ── Navigation: breadcrumb ─────────────────────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Input({"type": "breadcrumb-btn", "path": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_breadcrumb_click(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        triggered = ctx.triggered[0]
        if triggered["value"] is None or triggered["value"] == 0:
            return no_update, no_update, no_update
        prop_id = triggered["prop_id"]
        id_str = prop_id.rsplit(".", 1)[0]
        id_dict = json.loads(id_str)
        path = id_dict["path"]
        return "group", path, None

    # ── Navigation: variable row click ─────────────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Input({"type": "var-row", "path": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_variable_click(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        triggered = ctx.triggered[0]
        if triggered["value"] is None or triggered["value"] == 0:
            return no_update, no_update, no_update
        prop_id = triggered["prop_id"]
        id_str = prop_id.rsplit(".", 1)[0]
        id_dict = json.loads(id_str)
        varpath = id_dict["path"]

        # Check if this variable is clickable (different or almost_equal)
        for gr in state.group_results:
            for v in gr.variables:
                if v.varpath == varpath:
                    if v.status in ("different", "almost_equal"):
                        parent = "/" + "/".join(
                            varpath.strip("/").split("/")[:-1]
                        ) if "/" in varpath.strip("/") else "/"
                        return "detail", parent, varpath
                    return no_update, no_update, no_update

        return no_update, no_update, no_update

    # ── Navigation: back button ────────────────────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Input("back-to-group-btn", "n_clicks"),
        State("detail-parent-group", "data"),
        prevent_initial_call=True,
    )
    def on_back_click(n_clicks, parent_group):
        if not n_clicks:
            return no_update, no_update, no_update
        return "group", parent_group or "/", None

    # ── Navigation: previous variable ─────────────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Input("prev-var-btn", "n_clicks"),
        State("prev-varpath", "data"),
        prevent_initial_call=True,
    )
    def on_prev_var_click(n_clicks, prev_vp):
        if not n_clicks or not prev_vp:
            return no_update, no_update, no_update
        parent = ("/" + "/".join(prev_vp.strip("/").split("/")[:-1])
                  if "/" in prev_vp.strip("/") else "/")
        return "detail", parent, prev_vp

    # ── Root tabs: toggle Summary / Browse ──────────────────────

    app.clientside_callback(
        """
        function(nSummary, nBrowse) {
            var ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || ctx.triggered.length === 0)
                return [{'display': 'block'}, {'display': 'none'},
                        {border:'none',borderBottom:'3px solid #1565c0',
                         padding:'8px 20px',cursor:'pointer',fontSize:'13px',
                         fontWeight:'600',backgroundColor:'transparent',
                         color:'#1565c0'},
                        {border:'none',borderBottom:'3px solid transparent',
                         padding:'8px 20px',cursor:'pointer',fontSize:'13px',
                         fontWeight:'400',backgroundColor:'transparent',
                         color:'#78909c'}];

            var tid = ctx.triggered[0].prop_id;
            var showSummary = tid.indexOf('summary') !== -1;

            var activeStyle = {border:'none',borderBottom:'3px solid #1565c0',
                padding:'8px 20px',cursor:'pointer',fontSize:'13px',
                fontWeight:'600',backgroundColor:'transparent',
                color:'#1565c0'};
            var inactiveStyle = {border:'none',
                borderBottom:'3px solid transparent',
                padding:'8px 20px',cursor:'pointer',fontSize:'13px',
                fontWeight:'400',backgroundColor:'transparent',
                color:'#78909c'};

            if (showSummary) {
                return [{'display':'block'}, {'display':'none'},
                        activeStyle, inactiveStyle];
            } else {
                return [{'display':'none'}, {'display':'block'},
                        inactiveStyle, activeStyle];
            }
        }
        """,
        Output("root-panel-summary", "style"),
        Output("root-panel-browse", "style"),
        Output("root-tab-summary", "style"),
        Output("root-tab-browse", "style"),
        Input("root-tab-summary", "n_clicks"),
        Input("root-tab-browse", "n_clicks"),
        prevent_initial_call=True,
    )

    # ── Navigation: next variable ─────────────────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Input("next-var-btn", "n_clicks"),
        State("next-varpath", "data"),
        prevent_initial_call=True,
    )
    def on_next_var_click(n_clicks, next_vp):
        if not n_clicks or not next_vp:
            return no_update, no_update, no_update
        parent = ("/" + "/".join(next_vp.strip("/").split("/")[:-1])
                  if "/" in next_vp.strip("/") else "/")
        return "detail", parent, next_vp

    # ── Navigation: global summary variable row click ─────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Input({"type": "summary-var-row", "path": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_summary_var_click(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        triggered = ctx.triggered[0]
        if triggered["value"] is None or triggered["value"] == 0:
            return no_update, no_update, no_update
        prop_id = triggered["prop_id"]
        id_str = prop_id.rsplit(".", 1)[0]
        id_dict = json.loads(id_str)
        vp = id_dict["path"]
        parent = ("/" + "/".join(vp.strip("/").split("/")[:-1])
                  if "/" in vp.strip("/") else "/")
        return "detail", parent, vp

    # ── Search: filter variable table rows (clientside) ───────────

    app.clientside_callback(
        """
        function(searchValue) {
            var tbody = document.getElementById('var-table-body');
            if (!tbody) return window.dash_clientside.no_update;
            var rows = tbody.getElementsByTagName('tr');
            var term = (searchValue || '').toLowerCase();
            for (var i = 0; i < rows.length; i++) {
                var cells = rows[i].getElementsByTagName('td');
                if (cells.length > 0) {
                    var name = cells[0].textContent.toLowerCase();
                    rows[i].style.display = (term === '' || name.indexOf(term) !== -1)
                        ? '' : 'none';
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("var-search-input", "id"),
        Input("var-search-input", "value"),
        prevent_initial_call=True,
    )

    # ── Search: filter summary table rows (clientside) ─────────

    app.clientside_callback(
        """
        function(searchValue) {
            var tbody = document.getElementById('summary-table-body');
            if (!tbody) return window.dash_clientside.no_update;
            var rows = tbody.getElementsByTagName('tr');
            var term = (searchValue || '').toLowerCase();
            for (var i = 0; i < rows.length; i++) {
                var cells = rows[i].getElementsByTagName('td');
                if (cells.length > 0) {
                    var name = cells[0].textContent.toLowerCase();
                    rows[i].style.display = (term === '' || name.indexOf(term) !== -1)
                        ? '' : 'none';
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("summary-search-input", "id"),
        Input("summary-search-input", "value"),
        prevent_initial_call=True,
    )

    # ── Outlier: recompute outlier traces server-side ────────────

    @app.callback(
        Output({"type": "plot-graph", "label": ALL}, "figure",
               allow_duplicate=True),
        Output("outlier-count-display", "children"),
        Output("outlier-count-display", "style"),
        Input("outlier-sigma-slider", "value"),
        State("outlier-stats", "data"),
        State("nav-varpath", "data"),
        State({"type": "plot-graph", "label": ALL}, "id"),
        prevent_initial_call=True,
    )
    def update_outliers(n_sigma, stats, varpath, graph_ids):
        if not stats or not n_sigma or not varpath or not graph_ids:
            return ([no_update] * len(graph_ids), no_update, no_update)

        data_source = load_variable_data(state, varpath)
        d = data_source.source_data
        diff_arr = numpy.asarray(d["difference"], dtype=float)
        x_arr = numpy.asarray(d["x"])
        expected_arr = numpy.asarray(d["expected"], dtype=float)
        computed_arr = numpy.asarray(d["computed"], dtype=float)

        mean = stats["mean"]
        std = stats["std"]
        threshold = n_sigma * std
        outlier_mask = numpy.abs(diff_arr - mean) > threshold
        n_outliers = int(numpy.nansum(outlier_mask))

        # Find long_name
        long_name = varpath
        for gr in state.group_results:
            for v in gr.variables:
                if v.varpath == varpath:
                    long_name = v.long_name
                    break

        # Rebuild Signal Comparison and Signal Difference figures
        # with the new outlier threshold
        from .plots import (fig_signal_comparison, fig_signal_difference)
        new_sig_comp = fig_signal_comparison(long_name, data_source)
        new_sig_diff = fig_signal_difference(long_name, data_source)

        # Add outlier traces
        if numpy.any(outlier_mask):
            new_sig_comp.add_trace(go.Scattergl(
                x=x_arr[outlier_mask],
                y=expected_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "orange", "symbol": "diamond",
                            "line": {"width": 1, "color": "red"}},
                name="Reference (outlier)",
                hovertemplate=("index: %{x}<br>reference: %{y}<br>"
                               "<b>OUTLIER</b><extra></extra>"),
            ))
            new_sig_comp.add_trace(go.Scattergl(
                x=x_arr[outlier_mask],
                y=computed_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "yellow", "symbol": "diamond",
                            "line": {"width": 1, "color": "red"}},
                name="Target (outlier)",
                hovertemplate=("index: %{x}<br>target: %{y}<br>"
                               "<b>OUTLIER</b><extra></extra>"),
            ))
            new_sig_diff.add_trace(go.Scattergl(
                x=x_arr[outlier_mask],
                y=diff_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "orange", "symbol": "diamond",
                            "line": {"width": 1, "color": "red"}},
                name="Outlier",
                hovertemplate=("index: %{x}<br>difference: %{y}<br>"
                               "<b>OUTLIER</b><extra></extra>"),
            ))

        # Build results matching graph_ids order
        fig_map = {
            "Signal Comparison": new_sig_comp,
            "Signal Difference": new_sig_diff,
        }
        result = []
        for gid in graph_ids:
            label = gid["label"]
            if label in fig_map:
                result.append(fig_map[label])
            else:
                result.append(no_update)

        count_style = {
            "fontSize": "12px",
            "fontWeight": "600", "padding": "4px 12px",
            "borderRadius": "12px",
            "color": "#c62828" if n_outliers > 0 else "#2e7d32",
            "backgroundColor": "#ffcdd2" if n_outliers > 0 else "#c8e6c9",
        }
        return (result,
                f"Outliers detected: {n_outliers}",
                count_style)

    # ── Plot visibility toggle ─────────────────────────────────────

    @app.callback(
        Output({"type": "plot-container", "label": ALL}, "style"),
        Input("plot-visibility", "value"),
        State({"type": "plot-container", "label": ALL}, "id"),
        prevent_initial_call=True,
    )
    def toggle_plots(visible_labels, container_ids):
        styles = []
        for cid in container_ids:
            label = cid["label"]
            if label in visible_labels:
                styles.append({"marginBottom": "16px"})
            else:
                styles.append({"display": "none"})
        return styles

    # ── Marker size ────────────────────────────────────────────────
    # NOTE: Plotly figures in Dash are immutable once rendered.
    # We'd need to re-render figures to change marker size.
    # For now the slider updates are handled via a clientside callback
    # that patches the figure data directly.

    app.clientside_callback(
        """
        function(markerSize) {
            // Find all dcc.Graph components and update marker size
            var graphs = document.querySelectorAll('.js-plotly-plot');
            graphs.forEach(function(graphDiv) {
                if (graphDiv.data) {
                    var update = {};
                    for (var i = 0; i < graphDiv.data.length; i++) {
                        if (graphDiv.data[i].mode === 'markers') {
                            update['marker.size'] = markerSize;
                            Plotly.restyle(graphDiv, {'marker.size': markerSize});
                            break;
                        }
                    }
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("marker-size-slider", "id"),
        Input("marker-size-slider", "value"),
        prevent_initial_call=True,
    )

    # ── Zoom sync: Signal Comparison → other index-based graphs ──
    # Listen to ALL graphs' relayoutData, detect if Signal Comparison
    # triggered, then apply its x-range to all other index-based graphs.
    app.clientside_callback(
        """
        function(allRelayoutData, allIds) {
            // Guard against re-entrance from our own Plotly.relayout calls
            if (window._zoomSyncing) return window.dash_clientside.no_update;

            var ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || ctx.triggered.length === 0)
                return window.dash_clientside.no_update;

            // Find which graph triggered the callback
            var triggeredProp = ctx.triggered[0].prop_id;
            if (triggeredProp.indexOf('Signal Comparison') === -1)
                return window.dash_clientside.no_update;

            // Find the Signal Comparison index
            var triggerIdx = -1;
            for (var i = 0; i < allIds.length; i++) {
                if (allIds[i].label === 'Signal Comparison') {
                    triggerIdx = i;
                    break;
                }
            }
            if (triggerIdx === -1) return window.dash_clientside.no_update;

            var rd = allRelayoutData[triggerIdx];
            if (!rd) return window.dash_clientside.no_update;

            // Extract the x-range
            var xRange = null;
            var autorange = false;
            if (rd['xaxis.range[0]'] !== undefined) {
                xRange = [rd['xaxis.range[0]'], rd['xaxis.range[1]']];
            } else if (rd['xaxis.range'] !== undefined) {
                xRange = rd['xaxis.range'];
            } else if (rd['xaxis.autorange']) {
                autorange = true;
            } else {
                return window.dash_clientside.no_update;
            }

            // Apply to every other Plotly graph with x-axis "Index"
            window._zoomSyncing = true;
            var graphs = document.querySelectorAll('.js-plotly-plot');
            graphs.forEach(function(gd) {
                if (!gd.layout || !gd.layout.xaxis) return;
                var rs = gd.layout.xaxis.rangeslider;
                if (rs && rs.visible) return;
                var t = gd.layout.xaxis.title;
                var txt = (typeof t === 'string') ? t
                        : (t && t.text) ? t.text : '';
                if (txt !== 'Index') return;
                try {
                    if (autorange) {
                        Plotly.relayout(gd, {'xaxis.autorange': true});
                    } else {
                        Plotly.relayout(gd, {
                            'xaxis.range': [xRange[0], xRange[1]]
                        });
                    }
                } catch(e) {}
            });
            setTimeout(function() { window._zoomSyncing = false; }, 200);

            if (autorange) {
                return {auto: true};
            }
            return {min: xRange[0], max: xRange[1]};
        }
        """,
        Output("zoom-range", "data"),
        Input({"type": "plot-graph", "label": ALL}, "relayoutData"),
        State({"type": "plot-graph", "label": ALL}, "id"),
        prevent_initial_call=True,
    )

    # ── Zoom sync: update the geographical map ─────────────────────
    # Scattergeo cannot use xaxis.range; we filter the data points
    # to show only the ones within the zoomed index range.

    @app.callback(
        Output({"type": "plot-graph", "label": "Geographical Position"},
               "figure", allow_duplicate=True),
        Input("zoom-range", "data"),
        State("nav-varpath", "data"),
        prevent_initial_call=True,
    )
    def sync_map_zoom(zoom_range, varpath):
        if not zoom_range or not varpath:
            return no_update

        data_source = load_variable_data(state, varpath)
        fig = _build_filtered_map(data_source, zoom_range)
        return fig if fig is not None else no_update

    # ── Filter: variable hint (show min/max) ─────────────────────

    @app.callback(
        Output("filter-var-hint", "children"),
        Input("filter-var", "value"),
        prevent_initial_call=True,
    )
    def show_var_hint(varpath):
        if not varpath:
            return ""
        try:
            from .data_loading import _navigate_to_variable
            import numpy
            var = _navigate_to_variable(state.ref_ds, varpath)
            vals = var[:]
            if isinstance(vals, numpy.ma.MaskedArray):
                vals = vals.compressed()
            if len(vals) == 0:
                return "No valid values"
            vmin = numpy.nanmin(vals)
            vmax = numpy.nanmax(vals)
            return f"Range: {vmin:.6g} .. {vmax:.6g}  ({len(vals):,} pts)"
        except Exception:
            return ""

    # ── Filter: add condition ──────────────────────────────────────

    @app.callback(
        Output("filter-conditions", "data", allow_duplicate=True),
        Output("filter-val", "value"),
        Input("filter-add-btn", "n_clicks"),
        State("filter-var", "value"),
        State("filter-op", "value"),
        State("filter-val", "value"),
        State("filter-conditions", "data"),
        prevent_initial_call=True,
    )
    def add_condition(n_clicks, var, op, val, conditions):
        if not n_clicks or not var or not val:
            return no_update, no_update
        conditions = conditions or []
        conditions.append({"var": var, "op": op, "val": val.strip()})
        return conditions, ""

    # ── Filter: remove condition ───────────────────────────────────

    @app.callback(
        Output("filter-conditions", "data", allow_duplicate=True),
        Input({"type": "filter-remove", "index": ALL}, "n_clicks"),
        State("filter-conditions", "data"),
        prevent_initial_call=True,
    )
    def remove_condition(n_clicks_list, conditions):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        triggered = ctx.triggered[0]
        if triggered["value"] is None or triggered["value"] == 0:
            return no_update
        prop_id = triggered["prop_id"]
        id_str = prop_id.rsplit(".", 1)[0]
        id_dict = json.loads(id_str)
        idx = id_dict["index"]
        conditions = conditions or []
        if 0 <= idx < len(conditions):
            conditions.pop(idx)
        return conditions

    # ── Filter: display conditions + preview ───────────────────────

    @app.callback(
        Output("filter-conditions-display", "children"),
        Output("filter-preview", "children"),
        Output("filter-preview", "style"),
        Input("filter-conditions", "data"),
    )
    def display_conditions(conditions):
        conditions = conditions or []
        empty_preview_style = {
            "fontSize": "12px", "padding": "8px 12px",
            "borderRadius": "4px", "backgroundColor": "#f5f7fa",
            "color": "#607d8b",
        }

        if not conditions:
            return (
                html.Div("No conditions defined", style={
                    "fontSize": "11px", "color": "#bbb",
                    "fontStyle": "italic",
                }),
                "Add conditions above, then click Apply.",
                empty_preview_style,
            )

        # Render each condition as a removable pill
        pills = []
        for i, c in enumerate(conditions):
            short_var = c["var"].rsplit("/", 1)[-1]
            if c["op"] == "in":
                text = f"{short_var} in {c['val']}"
            else:
                text = f"{short_var} {c['op']} {c['val']}"
            pills.append(html.Div([
                html.Span(text, style={
                    "fontSize": "11px", "fontFamily": "monospace",
                }),
                html.Button(
                    "\u2715",
                    id={"type": "filter-remove", "index": i},
                    n_clicks=0,
                    style={
                        "border": "none", "background": "transparent",
                        "color": "#c62828", "cursor": "pointer",
                        "fontSize": "12px", "marginLeft": "6px",
                        "padding": "0 4px",
                    },
                ),
            ], style={
                "display": "inline-flex", "alignItems": "center",
                "padding": "3px 8px", "border": "1px solid #bbdefb",
                "borderRadius": "12px", "backgroundColor": "#e3f2fd",
                "marginRight": "4px", "marginBottom": "4px",
            }))

        # Compute preview: how many points would be masked
        expr = _conditions_to_expression(conditions)
        rewritten = rewrite_expression(expr)
        mask, error = create_expression_mask(state.ref_ds, rewritten)
        if mask is None:
            preview_text = f"Error: {error}"
            preview_style = {
                "fontSize": "12px", "padding": "8px 12px",
                "borderRadius": "4px", "backgroundColor": "#ffcdd2",
                "color": "#c62828",
            }
        else:
            total = int(mask.size)
            selected = int(mask.sum())
            masked = total - selected
            pct = 100 * masked / total if total > 0 else 0
            preview_text = (
                f"Preview: {selected:,} / {total:,} points kept "
                f"({masked:,} masked, {pct:.1f}%)")
            preview_style = {
                "fontSize": "12px", "padding": "8px 12px",
                "borderRadius": "4px", "backgroundColor": "#e8f5e9",
                "color": "#2e7d32",
            }

        return (
            html.Div(pills, style={
                "display": "flex", "flexWrap": "wrap"}),
            preview_text,
            preview_style,
        )

    # ── Filter: apply ──────────────────────────────────────────────
    # Update state, then directly re-render main content.

    @app.callback(
        Output("expr-status", "children"),
        Output("expr-status", "style"),
        Output("expr-textarea", "value"),
        Output("main-content", "children", allow_duplicate=True),
        Output("filter-banner", "children", allow_duplicate=True),
        Output("filter-banner", "style", allow_duplicate=True),
        Input("filter-apply-btn", "n_clicks"),
        State("filter-conditions", "data"),
        prevent_initial_call=True,
    )
    def apply_filter(n_clicks, conditions):
        if not n_clicks:
            return (no_update,) * 6
        conditions = conditions or []
        if not conditions:
            return (no_update,) * 6

        raw = _conditions_to_expression(conditions)
        result = _apply_expression_no_nav(state, raw)
        banner = _make_banner(state)
        return (*result, raw, _render_current_view(state), *banner)

    # ── Filter: clear all ──────────────────────────────────────────

    @app.callback(
        Output("filter-conditions", "data", allow_duplicate=True),
        Output("expr-textarea", "value", allow_duplicate=True),
        Output("expr-status", "children", allow_duplicate=True),
        Output("expr-status", "style", allow_duplicate=True),
        Output("main-content", "children", allow_duplicate=True),
        Output("filter-banner", "children", allow_duplicate=True),
        Output("filter-banner", "style", allow_duplicate=True),
        Input("filter-clear-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_filter(n_clicks):
        if not n_clicks:
            return (no_update,) * 7
        state.active_mask = None
        state.active_expression = None
        state.filter_info = None
        state.clear_variable_cache()
        state.group_results = scan_all_groups(state.ref_ds, state.tgt_ds)
        return ([], "", "No filter active", _STATUS_NONE,
                _render_current_view(state), *_make_banner(state))

    # ── Advanced mode: apply raw expression ────────────────────────

    @app.callback(
        Output("expr-status", "children", allow_duplicate=True),
        Output("expr-status", "style", allow_duplicate=True),
        Output("main-content", "children", allow_duplicate=True),
        Output("filter-banner", "children", allow_duplicate=True),
        Output("filter-banner", "style", allow_duplicate=True),
        Input("expr-apply-btn", "n_clicks"),
        State("expr-textarea", "value"),
        prevent_initial_call=True,
    )
    def apply_advanced_expression(n_clicks, expr_text):
        if not n_clicks:
            return (no_update,) * 5
        raw = (expr_text or "").strip()
        if not raw:
            return (no_update,) * 5
        result = _apply_expression_no_nav(state, raw)
        return (*result, _render_current_view(state), *_make_banner(state))

    # ── Flag filter ────────────────────────────────────────────────
    # Update graph figures directly — no page re-render so the
    # dropdown keeps its selection state.

    @app.callback(
        Output({"type": "plot-graph", "label": ALL}, "figure",
               allow_duplicate=True),
        Output("outlier-count-display", "children", allow_duplicate=True),
        Output("outlier-count-display", "style", allow_duplicate=True),
        Output("outlier-stats", "data", allow_duplicate=True),
        Input("flag-dropdown", "value"),
        State("nav-varpath", "data"),
        State({"type": "plot-graph", "label": ALL}, "id"),
        State("outlier-sigma-slider", "value"),
        prevent_initial_call=True,
    )
    def apply_flag_filter(selected_flags, varpath, graph_ids, n_sigma):
        import numpy
        n_graphs = len(graph_ids) if graph_ids else 0
        if not varpath or not graph_ids:
            return ([no_update] * n_graphs, no_update, no_update, no_update)

        try:
            data_source = load_variable_data(state, varpath)
        except Exception:
            return ([no_update] * n_graphs, no_update, no_update, no_update)

        if not data_source.flags:
            return ([no_update] * n_graphs, no_update, no_update, no_update)

        # Apply flag filter on the DataSource
        if not selected_flags:
            data_source.apply_flag_filter(
                numpy.arange(len(data_source.full_x)))
        else:
            indices = data_source.flags[0].compute_filtered_indices(
                selected_flags, data_source.flags)
            data_source.apply_flag_filter(indices)

        # Find the long name for rebuilding figures
        long_name = varpath
        for gr in state.group_results:
            for v in gr.variables:
                if v.varpath == varpath:
                    long_name = v.long_name
                    break

        # Rebuild all figures from updated data
        new_configs = build_all_figures(long_name, data_source)
        new_fig_map = dict(new_configs)

        # Recompute outliers on filtered data
        d = data_source.source_data
        diff_arr = numpy.asarray(d["difference"], dtype=float)
        valid_diff = diff_arr[~numpy.isnan(diff_arr)]
        outlier_mean = float(numpy.mean(valid_diff)) if len(valid_diff) > 0 else 0.0
        outlier_std = float(numpy.std(valid_diff)) if len(valid_diff) > 0 else 1.0
        if not n_sigma:
            n_sigma = 3
        outlier_mask = numpy.abs(diff_arr - outlier_mean) > (n_sigma * outlier_std)
        n_outliers = int(numpy.nansum(outlier_mask))

        x_arr = numpy.asarray(d["x"])
        expected_arr = numpy.asarray(d["expected"], dtype=float)
        computed_arr = numpy.asarray(d["computed"], dtype=float)

        if numpy.any(outlier_mask) and "Signal Comparison" in new_fig_map:
            new_fig_map["Signal Comparison"].add_trace(go.Scattergl(
                x=x_arr[outlier_mask], y=expected_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "orange", "symbol": "diamond",
                            "line": {"width": 1, "color": "red"}},
                name="Reference (outlier)",
                hovertemplate=("index: %{x}<br>reference: %{y}<br>"
                               "<b>OUTLIER</b><extra></extra>"),
            ))
            new_fig_map["Signal Comparison"].add_trace(go.Scattergl(
                x=x_arr[outlier_mask], y=computed_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "yellow", "symbol": "diamond",
                            "line": {"width": 1, "color": "red"}},
                name="Target (outlier)",
                hovertemplate=("index: %{x}<br>target: %{y}<br>"
                               "<b>OUTLIER</b><extra></extra>"),
            ))
        if numpy.any(outlier_mask) and "Signal Difference" in new_fig_map:
            new_fig_map["Signal Difference"].add_trace(go.Scattergl(
                x=x_arr[outlier_mask], y=diff_arr[outlier_mask],
                mode="markers",
                marker={"size": 6, "color": "orange", "symbol": "diamond",
                            "line": {"width": 1, "color": "red"}},
                name="Outlier",
                hovertemplate=("index: %{x}<br>difference: %{y}<br>"
                               "<b>OUTLIER</b><extra></extra>"),
            ))

        # Return figures in the same order as graph_ids
        result = []
        for gid in graph_ids:
            label = gid["label"]
            if label in new_fig_map:
                result.append(new_fig_map[label])
            else:
                result.append(no_update)

        count_style = {
            "fontSize": "12px", "fontWeight": "600", "padding": "4px 12px",
            "borderRadius": "12px",
            "color": "#c62828" if n_outliers > 0 else "#2e7d32",
            "backgroundColor": "#ffcdd2" if n_outliers > 0 else "#c8e6c9",
        }
        return (result,
                f"Outliers detected: {n_outliers}",
                count_style,
                {"mean": outlier_mean, "std": outlier_std})

    # ── HTML export ────────────────────────────────────────────────

    @app.callback(
        Output("export-download", "data"),
        Input("export-btn", "n_clicks"),
        State("nav-varpath", "data"),
        prevent_initial_call=True,
    )
    def export_html(n_clicks, varpath):
        if not n_clicks:
            return no_update

        if not varpath:
            return {
                "content": "<h1>No variable selected</h1>"
                "<p>Select a variable with differences to export.</p>",
                "filename": "comparison_export.html",
                "type": "text/html",
            }

        # Find variable info
        long_name = varpath
        for gr in state.group_results:
            for v in gr.variables:
                if v.varpath == varpath:
                    long_name = v.long_name
                    break

        data_source = load_variable_data(state, varpath)
        plot_configs = build_all_figures(long_name, data_source)

        # Convert numpy arrays to plain Python lists so JSON
        # serialisation embeds the actual data in the HTML file.
        import numpy
        for _label, fig in plot_configs:
            for trace in fig.data:
                for attr in ("x", "y", "width"):
                    val = getattr(trace, attr, None)
                    if val is not None:
                        if isinstance(val, numpy.ndarray):
                            setattr(trace, attr, val.tolist())
                        elif isinstance(val, list) and val and isinstance(
                                val[0], (numpy.integer, numpy.floating)):
                            setattr(trace, attr, [v.item() for v in val])

        # Build a combined HTML — first figure includes plotly.js,
        # subsequent ones reuse it.
        html_parts = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Comparison: {varpath}</title>",
            "<style>body{font-family:'Segoe UI',sans-serif;margin:20px;}"
            "h2{color:#1565c0;}h3{color:#455a64;}</style>",
            "</head><body>",
            f"<h2>Comparison: {varpath}</h2>",
            f"<p>{long_name}</p>",
            f"<p>Reference: {state.ref_path.name} | "
            f"Target: {state.tgt_path.name}</p>",
        ]
        for i, (label, fig) in enumerate(plot_configs):
            html_parts.append(f"<h3>{label}</h3>")
            html_parts.append(pio.to_html(
                fig,
                include_plotlyjs="cdn" if i == 0 else False,
                full_html=False,
            ))
        html_parts.append("</body></html>")

        return {
            "content": "".join(html_parts),
            "filename": "comparison_export.html",
            "type": "text/html",
        }

    # ── Share: encode state into URL ──────────────────────────────
    # Clicking "Share" writes the current state as URL search params.
    # A clientside callback copies the URL to clipboard and shows a
    # toast notification.

    @app.callback(
        Output("url-location", "search"),
        Output("share-toast", "children"),
        Output("share-toast", "style"),
        Input("share-btn", "n_clicks"),
        State("nav-current-view", "data"),
        State("nav-group-path", "data"),
        State("nav-varpath", "data"),
        State("zoom-range", "data"),
        State("filter-conditions", "data"),
        State("expr-textarea", "value"),
        prevent_initial_call=True,
    )
    def share_state(n_clicks, view, group_path, varpath,
                    zoom_range, conditions, expr_text):
        if not n_clicks:
            return no_update, no_update, no_update
        import urllib.parse
        params = {}
        params["view"] = view or "group"
        if group_path and group_path != "/":
            params["group"] = group_path
        if varpath:
            params["var"] = varpath
        if zoom_range:
            params["zoom"] = json.dumps(zoom_range)
        if conditions:
            params["filter"] = json.dumps(conditions)
        if expr_text and expr_text.strip():
            params["expr"] = expr_text.strip()

        search = "?" + urllib.parse.urlencode(params)
        toast_style = {
            "backgroundColor": "rgba(255,255,255,0.9)",
            "color": "#1565c0", "padding": "4px 10px",
            "borderRadius": "4px", "fontSize": "11px",
            "fontWeight": "600", "whiteSpace": "nowrap",
        }
        return search, "URL copied!", toast_style

    # Copy URL to clipboard when search params change
    app.clientside_callback(
        """
        function(search) {
            if (!search) return '';
            var url = window.location.origin + window.location.pathname
                      + search;
            navigator.clipboard.writeText(url).catch(function() {});
            // Hide toast after 3 seconds
            setTimeout(function() {
                var el = document.getElementById('share-toast');
                if (el) el.style.display = 'none';
            }, 3000);
            return '';
        }
        """,
        Output("share-toast", "id"),
        Input("url-location", "search"),
        prevent_initial_call=True,
    )

    # ── Restore state from URL on initial load ────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Output("nav-group-path", "data", allow_duplicate=True),
        Output("nav-varpath", "data", allow_duplicate=True),
        Output("filter-conditions", "data", allow_duplicate=True),
        Output("expr-textarea", "value", allow_duplicate=True),
        Output("expr-status", "children", allow_duplicate=True),
        Output("expr-status", "style", allow_duplicate=True),
        Output("main-content", "children", allow_duplicate=True),
        Output("filter-banner", "children", allow_duplicate=True),
        Output("filter-banner", "style", allow_duplicate=True),
        Output("url-initial-load", "data"),
        Output("zoom-range", "data", allow_duplicate=True),
        Input("url-location", "search"),
        State("url-initial-load", "data"),
        prevent_initial_call=True,
    )
    def restore_from_url(search, is_initial):
        n_out = 12
        if not is_initial or not search:
            return (no_update,) * n_out
        import urllib.parse
        params = dict(urllib.parse.parse_qsl(
            search.lstrip("?")))
        if not params:
            return (no_update,) * n_out

        view = params.get("view", "group")
        group_path = params.get("group", "/")
        varpath = params.get("var")
        filter_json = params.get("filter")
        expr_text = params.get("expr", "")
        zoom_json = params.get("zoom")

        # Restore conditions
        conditions = []
        if filter_json:
            try:
                conditions = json.loads(filter_json)
            except (json.JSONDecodeError, TypeError):
                pass

        # Restore zoom range
        zoom_range = no_update
        if zoom_json:
            try:
                zoom_range = json.loads(zoom_json)
            except (json.JSONDecodeError, TypeError):
                pass

        # Apply expression filter if present
        raw_expr = expr_text
        if conditions and not raw_expr:
            raw_expr = _conditions_to_expression(conditions)

        status_msg = "No filter active"
        status_style = _STATUS_NONE
        if raw_expr:
            result = _apply_expression_no_nav(state, raw_expr)
            if result:
                status_msg, status_style = result

        # Set navigation state
        if view == "detail" and varpath:
            state.current_view = "detail"
            state.current_varpath = varpath
        else:
            state.current_view = "group"
            state.current_group_path = group_path

        banner = _make_banner(state)
        # Parse zoom_range for figure construction
        restored_zoom = None
        if zoom_range is not no_update:
            restored_zoom = zoom_range
        content = _render_current_view(state, zoom_range=restored_zoom)

        return (view, group_path, varpath,
                conditions, expr_text,
                status_msg, status_style,
                content, *banner,
                False, zoom_range)
