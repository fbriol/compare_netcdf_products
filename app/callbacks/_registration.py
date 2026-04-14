"""Dash callbacks for the NetCDF comparison application."""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING

import numpy
import plotly.graph_objects as go
import plotly.io as pio
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback_context,
    html,
    no_update,
)

if TYPE_CHECKING:
    import dash

    from ..state import SessionState

from ..comparison import FilterInfo, rescan_affected_groups, scan_all_groups
from ..data_loading import load_variable_data
from ..expression import (
    create_expression_mask,
    find_affected_groups,
    rewrite_expression,
)
from ..plots import build_all_figures
from ..utils import (
    count_diffs,
    count_diffs_recursive,
    find_group_result,
    navigate_to_group,
)
from .views import (
    build_filtered_map,
    render_current_view,
)

LOGGER = logging.getLogger(__name__)


# Backwards-compatible aliases (used internally with underscore prefix)
_find_group_result = find_group_result
_navigate_to_group = navigate_to_group
_count_diffs = count_diffs
_count_diffs_recursive = count_diffs_recursive


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
    "fontSize": "12px",
    "padding": "8px 12px",
    "borderRadius": "4px",
    "backgroundColor": "#c8e6c9",
    "color": "#2e7d32",
}
_STATUS_ERR = {
    "fontSize": "12px",
    "padding": "8px 12px",
    "borderRadius": "4px",
    "backgroundColor": "#ffcdd2",
    "color": "#c62828",
}
_STATUS_NONE = {
    "fontSize": "12px",
    "padding": "8px 12px",
    "borderRadius": "4px",
    "backgroundColor": "#e3f2fd",
    "color": "#1565c0",
}

_BANNER_ACTIVE = {
    "backgroundColor": "#ff9800",
    "color": "white",
    "padding": "4px 14px",
    "borderRadius": "14px",
    "fontSize": "12px",
    "fontWeight": "600",
    "letterSpacing": "0.3px",
    "display": "inline-flex",
    "alignItems": "center",
    "gap": "6px",
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
    mask, error = create_expression_mask(state.src_ds, rewritten)
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
    LOGGER.info(
        "_apply_expression_no_nav: %d/%d selected, mask id=%d",
        selected,
        total,
        id(mask),
    )
    state.filter_info = FilterInfo(
        expression=raw,
        selected_points=selected,
        total_points=total,
        affected_groups=affected,
    )
    state.group_results = rescan_affected_groups(
        state.src_ds,
        state.tgt_ds,
        state.group_results,
        state.filter_info,
        mask=state.active_mask,
    )
    msg = (
        f"Filter active: {masked:,}/{total:,} points masked "
        f"({100 * masked / total:.1f}%)"
    )
    return (msg, _STATUS_OK)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  View builders → moved to callbacks/views.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Callback registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def register_callbacks(app: dash.Dash, state: SessionState) -> None:
    """Register all Dash callbacks.

    Args:
        app: The Dash application.
        state: The shared session state.

    """

    # ── Sidebar toggle ──────────────────────────────────────────────
    app.clientside_callback(
        """
        function(n) {
            var sb = document.getElementById('sidebar');
            if (!sb) return window.dash_clientside.no_update;
            if (sb.style.display === 'none') {
                sb.style.display = '';
                sb.style.marginLeft = '0';
                sb.style.opacity = '1';
            } else {
                sb.style.display = 'none';
            }
            // Trigger a window resize so Plotly graphs resize
            setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 350);
            return window.dash_clientside.no_update;
        }
        """,
        Output("sidebar-toggle-btn", "id"),
        Input("sidebar-toggle-btn", "n_clicks"),
        prevent_initial_call=True,
    )

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
        return render_current_view(state, zoom_range=zoom_range)

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
        best_val, best_id = 0, None
        for t in ctx.triggered:
            val = t.get("value") or 0
            if val > best_val:
                best_val = val
                best_id = json.loads(t["prop_id"].rsplit(".", 1)[0])
        if best_id is None or best_val == 0:
            return no_update, no_update, no_update
        return "group", best_id["path"], None

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
        best_val, best_id = 0, None
        for t in ctx.triggered:
            val = t.get("value") or 0
            if val > best_val:
                best_val = val
                best_id = json.loads(t["prop_id"].rsplit(".", 1)[0])
        if best_id is None or best_val == 0:
            return no_update, no_update, no_update
        path = best_id["path"]
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
        # Find the actually clicked row (highest n_clicks among triggered)
        best_val = 0
        best_id = None
        for t in ctx.triggered:
            val = t.get("value") or 0
            if val > best_val:
                best_val = val
                prop_id = t["prop_id"]
                id_str = prop_id.rsplit(".", 1)[0]
                best_id = json.loads(id_str)
        if best_id is None or best_val == 0:
            return no_update, no_update, no_update
        varpath = best_id["path"]

        parent = (
            "/" + "/".join(varpath.strip("/").split("/")[:-1])
            if "/" in varpath.strip("/")
            else "/"
        )
        return "detail", parent, varpath

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
        parent = (
            "/" + "/".join(prev_vp.strip("/").split("/")[:-1])
            if "/" in prev_vp.strip("/")
            else "/"
        )
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
        parent = (
            "/" + "/".join(next_vp.strip("/").split("/")[:-1])
            if "/" in next_vp.strip("/")
            else "/"
        )
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
        best_val, best_id = 0, None
        for t in ctx.triggered:
            val = t.get("value") or 0
            if val > best_val:
                best_val = val
                best_id = json.loads(t["prop_id"].rsplit(".", 1)[0])
        if best_id is None or best_val == 0:
            return no_update, no_update, no_update
        vp = best_id["path"]
        parent = (
            "/" + "/".join(vp.strip("/").split("/")[:-1])
            if "/" in vp.strip("/")
            else "/"
        )
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
        Output({"type": "plot-graph", "label": ALL}, "figure", allow_duplicate=True),
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
        from ..plots import fig_signal_comparison, fig_signal_difference

        new_sig_comp = fig_signal_comparison(long_name, data_source)
        new_sig_diff = fig_signal_difference(long_name, data_source)

        # Add outlier traces
        if numpy.any(outlier_mask):
            new_sig_comp.add_trace(
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
                    hovertemplate=(
                        "index: %{x}<br>source: %{y}<br><b>OUTLIER</b><extra></extra>"
                    ),
                )
            )
            new_sig_comp.add_trace(
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
                    hovertemplate=(
                        "index: %{x}<br>target: %{y}<br><b>OUTLIER</b><extra></extra>"
                    ),
                )
            )
            new_sig_diff.add_trace(
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
                    hovertemplate=(
                        "index: %{x}<br>difference: %{y}<br>"
                        "<b>OUTLIER</b><extra></extra>"
                    ),
                )
            )

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
            "fontWeight": "600",
            "padding": "4px 12px",
            "borderRadius": "12px",
            "color": "#c62828" if n_outliers > 0 else "#2e7d32",
            "backgroundColor": "#ffcdd2" if n_outliers > 0 else "#c8e6c9",
        }
        return (result, f"Outliers detected: {n_outliers}", count_style)

    # ── Plot visibility toggle (clientside — no server round-trip) ──

    app.clientside_callback(
        """
        function(visibleLabels, containerIds) {
            var styles = [];
            for (var i = 0; i < containerIds.length; i++) {
                var label = containerIds[i].label;
                if (visibleLabels.indexOf(label) !== -1) {
                    styles.push({"marginBottom": "16px"});
                } else {
                    styles.push({"display": "none"});
                }
            }
            return styles;
        }
        """,
        Output({"type": "plot-container", "label": ALL}, "style"),
        Input("plot-visibility", "value"),
        State({"type": "plot-container", "label": ALL}, "id"),
        prevent_initial_call=True,
    )

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

    # ── Zoom sync: any index-based graph → all other index-based graphs ──
    # Uses native Plotly plotly_relayout events for reliable sync,
    # installed via MutationObserver. The Dash callback only updates
    # the zoom-range dcc.Store for server-side use (geo map rebuild).
    app.clientside_callback(
        """
        function(n) {
            // Install zoom-sync event listeners once
            if (window._zoomSyncInstalled) return window.dash_clientside.no_update;
            window._zoomSyncInstalled = true;
            window._zoomSyncing = false;
            window._lastZoomRange = null;

            function isIndexGraph(gd) {
                if (!gd.layout || !gd.layout.xaxis) return false;
                var t = gd.layout.xaxis.title;
                var txt = (typeof t === 'string') ? t
                        : (t && t.text) ? t.text : '';
                return txt === 'Index';
            }

            function attachListeners() {
                var graphs = document.querySelectorAll('.js-plotly-plot');
                graphs.forEach(function(gd) {
                    if (gd._zoomSyncAttached) return;
                    gd._zoomSyncAttached = true;
                    gd.on('plotly_relayout', function(ed) {
                        if (window._zoomSyncing) return;
                        if (!ed) return;
                        if (!isIndexGraph(gd)) return;

                        var xRange = null;
                        var autorange = false;
                        if (ed['xaxis.range[0]'] !== undefined) {
                            xRange = [ed['xaxis.range[0]'],
                                      ed['xaxis.range[1]']];
                        } else if (ed['xaxis.range'] !== undefined) {
                            xRange = ed['xaxis.range'];
                        } else if (ed['xaxis.autorange']) {
                            autorange = true;
                        } else {
                            return;
                        }

                        window._zoomSyncing = true;
                        document.querySelectorAll('.js-plotly-plot')
                            .forEach(function(other) {
                                if (other === gd) return;
                                if (!isIndexGraph(other)) return;
                                try {
                                    if (autorange) {
                                        Plotly.relayout(other,
                                            {'xaxis.autorange': true});
                                    } else {
                                        Plotly.relayout(other, {
                                            'xaxis.range[0]': xRange[0],
                                            'xaxis.range[1]': xRange[1]
                                        });
                                    }
                                } catch(e) {}
                            });
                        setTimeout(function() {
                            window._zoomSyncing = false;
                        }, 300);

                        // Store for server-side access
                        if (autorange) {
                            window._lastZoomRange = {auto: true};
                        } else {
                            window._lastZoomRange = {
                                min: xRange[0], max: xRange[1]};
                        }
                    });
                });
            }

            // Attach now and on future DOM changes
            setTimeout(attachListeners, 300);
            var obs = new MutationObserver(function() {
                setTimeout(attachListeners, 200);
            });
            obs.observe(document.body, {childList: true, subtree: true});

            return window.dash_clientside.no_update;
        }
        """,
        Output("zoom-range", "data"),
        Input("main-content", "children"),
        prevent_initial_call=True,
    )

    # Update zoom-range store from relayoutData changes
    app.clientside_callback(
        """
        function(allRelayoutData) {
            // Return the stored zoom range from the native event handler
            if (window._lastZoomRange) {
                var r = window._lastZoomRange;
                window._lastZoomRange = null;
                return r;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("zoom-range", "data", allow_duplicate=True),
        Input({"type": "plot-graph", "label": ALL}, "relayoutData"),
        prevent_initial_call=True,
    )

    # ── Zoom sync: update the geographical map ─────────────────────
    # Scattergeo cannot use xaxis.range; we filter the data points
    # to show only the ones within the zoomed index range.

    @app.callback(
        Output(
            {"type": "plot-graph", "label": "Geographical Position"},
            "figure",
            allow_duplicate=True,
        ),
        Input("zoom-range", "data"),
        State("nav-varpath", "data"),
        prevent_initial_call=True,
    )
    def sync_map_zoom(zoom_range, varpath):
        if not zoom_range or not varpath:
            return no_update

        data_source = load_variable_data(state, varpath)
        fig = build_filtered_map(data_source, zoom_range)
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
            import numpy

            from ..data_loading import _navigate_to_variable

            var = _navigate_to_variable(state.src_ds, varpath)
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
            "fontSize": "12px",
            "padding": "8px 12px",
            "borderRadius": "4px",
            "backgroundColor": "#f5f7fa",
            "color": "#607d8b",
        }

        if not conditions:
            return (
                html.Div(
                    "No conditions defined",
                    style={
                        "fontSize": "11px",
                        "color": "#bbb",
                        "fontStyle": "italic",
                    },
                ),
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
            pills.append(
                html.Div(
                    [
                        html.Span(
                            text,
                            style={
                                "fontSize": "11px",
                                "fontFamily": "monospace",
                            },
                        ),
                        html.Button(
                            "\u2715",
                            id={"type": "filter-remove", "index": i},
                            n_clicks=0,
                            style={
                                "border": "none",
                                "background": "transparent",
                                "color": "#c62828",
                                "cursor": "pointer",
                                "fontSize": "12px",
                                "marginLeft": "6px",
                                "padding": "0 4px",
                            },
                        ),
                    ],
                    style={
                        "display": "inline-flex",
                        "alignItems": "center",
                        "padding": "3px 8px",
                        "border": "1px solid #bbdefb",
                        "borderRadius": "12px",
                        "backgroundColor": "#e3f2fd",
                        "marginRight": "4px",
                        "marginBottom": "4px",
                    },
                )
            )

        # Compute preview: how many points would be masked
        expr = _conditions_to_expression(conditions)
        rewritten = rewrite_expression(expr)
        mask, error = create_expression_mask(state.src_ds, rewritten)
        if mask is None:
            preview_text = f"Error: {error}"
            preview_style = {
                "fontSize": "12px",
                "padding": "8px 12px",
                "borderRadius": "4px",
                "backgroundColor": "#ffcdd2",
                "color": "#c62828",
            }
        else:
            total = int(mask.size)
            selected = int(mask.sum())
            masked = total - selected
            pct = 100 * masked / total if total > 0 else 0
            preview_text = (
                f"Preview: {selected:,} / {total:,} points kept "
                f"({masked:,} masked, {pct:.1f}%)"
            )
            preview_style = {
                "fontSize": "12px",
                "padding": "8px 12px",
                "borderRadius": "4px",
                "backgroundColor": "#e8f5e9",
                "color": "#2e7d32",
            }

        return (
            html.Div(pills, style={"display": "flex", "flexWrap": "wrap"}),
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
        return (*result, raw, render_current_view(state), *banner)

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
        state.group_results = scan_all_groups(state.src_ds, state.tgt_ds)
        return (
            [],
            "",
            "No filter active",
            _STATUS_NONE,
            render_current_view(state),
            *_make_banner(state),
        )

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
        return (*result, render_current_view(state), *_make_banner(state))

    # ── Flag filter ────────────────────────────────────────────────
    # Update graph figures directly — no page re-render so the
    # dropdown keeps its selection state.

    @app.callback(
        Output({"type": "plot-graph", "label": ALL}, "figure", allow_duplicate=True),
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
            data_source.apply_flag_filter(numpy.arange(len(data_source.full_x)))
        else:
            indices = data_source.flags[0].compute_filtered_indices(
                selected_flags, data_source.flags
            )
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
            new_fig_map["Signal Comparison"].add_trace(
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
                    hovertemplate=(
                        "index: %{x}<br>source: %{y}<br><b>OUTLIER</b><extra></extra>"
                    ),
                )
            )
            new_fig_map["Signal Comparison"].add_trace(
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
                    hovertemplate=(
                        "index: %{x}<br>target: %{y}<br><b>OUTLIER</b><extra></extra>"
                    ),
                )
            )
        if numpy.any(outlier_mask) and "Signal Difference" in new_fig_map:
            new_fig_map["Signal Difference"].add_trace(
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
                    hovertemplate=(
                        "index: %{x}<br>difference: %{y}<br>"
                        "<b>OUTLIER</b><extra></extra>"
                    ),
                )
            )

        # Return figures in the same order as graph_ids
        result = []
        for gid in graph_ids:
            label = gid["label"]
            if label in new_fig_map:
                result.append(new_fig_map[label])
            else:
                result.append(no_update)

        count_style = {
            "fontSize": "12px",
            "fontWeight": "600",
            "padding": "4px 12px",
            "borderRadius": "12px",
            "color": "#c62828" if n_outliers > 0 else "#2e7d32",
            "backgroundColor": "#ffcdd2" if n_outliers > 0 else "#c8e6c9",
        }
        return (
            result,
            f"Outliers detected: {n_outliers}",
            count_style,
            {"mean": outlier_mean, "std": outlier_std},
        )

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
                        elif (
                            isinstance(val, list)
                            and val
                            and isinstance(val[0], (numpy.integer, numpy.floating))
                        ):
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
            f"<p>Reference: {state.src_path.name} | Target: {state.tgt_path.name}</p>",
        ]
        for i, (label, fig) in enumerate(plot_configs):
            html_parts.append(f"<h3>{label}</h3>")
            html_parts.append(
                pio.to_html(
                    fig,
                    include_plotlyjs="cdn" if i == 0 else False,
                    full_html=False,
                )
            )
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
    def share_state(
        n_clicks, view, group_path, varpath, zoom_range, conditions, expr_text
    ):
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
            "color": "#1565c0",
            "padding": "4px 10px",
            "borderRadius": "4px",
            "fontSize": "11px",
            "fontWeight": "600",
            "whiteSpace": "nowrap",
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

        params = dict(urllib.parse.parse_qsl(search.lstrip("?")))
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
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                conditions = json.loads(filter_json)

        # Restore zoom range
        zoom_range = no_update
        if zoom_json:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                zoom_range = json.loads(zoom_json)

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
        content = render_current_view(state, zoom_range=restored_zoom)

        return (
            view,
            group_path,
            varpath,
            conditions,
            expr_text,
            status_msg,
            status_style,
            content,
            *banner,
            False,
            zoom_range,
        )
