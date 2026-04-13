"""Panel widgets for the NetCDF comparison application.

This module builds the complete UI:

* **Group browser** -- Main-area view showing attributes, sub-groups and
  variables for a selected group.
* **Detail view** -- Main-area view showing Bokeh comparison plots for a
  single variable.
* **Expression panel** -- Free-text expression builder with ``${...}``
  variable references and auto-completion.
* **Legend / Export** -- Colour legend and HTML export button.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
import fnmatch
import io
import logging

import pandas as pd
import panel as pn

if TYPE_CHECKING:
    from .state import SessionState
    import netCDF4

    from .comparison import (
        AttributeComparisonResult,
        GroupComparisonResult,
        VariableComparisonResult,
    )
    from .data_loading import DataSource

from .comparison import FilterInfo, scan_all_groups
from .constants import LEGEND_DATA, STATUS_COLORS, STATUS_TEXT_COLORS
from .data_loading import load_variable_data
from .expression import (
    create_expression_mask,
    find_affected_groups,
    list_all_variables,
    rewrite_expression,
)
from .plots import build_dashboard

LOGGER = logging.getLogger(__name__)

# ── Shared CSS ──────────────────────────────────────────────────────

_CARD_CSS = """
:host {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 8px;
  background: #ffffff;
}
"""

_NAV_BTN_CSS = """
:host .bk-btn {
  text-align: left;
  font-size: 12px;
  padding: 5px 10px;
  border: none;
  border-radius: 4px;
  transition: background-color 0.15s;
}
:host .bk-btn:hover {
  background-color: #e3f2fd;
}
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _find_group_result(
    state: SessionState,
    group_path: str,
) -> GroupComparisonResult | None:
    """Find the comparison result for *group_path*."""
    for gr in state.group_results:
        if gr.group_path == group_path:
            return gr
    return None


def _count_diffs(state: SessionState, group_path: str) -> tuple[int, int]:
    """Return (n_different, n_total) variables in *group_path*."""
    gr = _find_group_result(state, group_path)
    if gr is None:
        return 0, 0
    n_diff = sum(1 for v in gr.variables
                 if v.status in ("different", "almost_equal"))
    return n_diff, len(gr.variables)


def _count_diffs_recursive(
    state: SessionState,
    dataset: netCDF4.Dataset,
    group_path: str,
) -> tuple[int, int]:
    """Return (n_different, n_total) across *group_path* and children."""
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


def _navigate_to_group(
    dataset: netCDF4.Dataset,
    group_path: str,
) -> netCDF4.Group | None:
    """Navigate to a group by path."""
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Breadcrumb
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_breadcrumb(
    path: str,
    on_navigate: Callable[[str], None],
) -> pn.Row:
    """Build a clickable breadcrumb bar."""
    parts = [p for p in path.split("/") if p]
    items: list[pn.viewable.Viewable] = []

    root_btn = pn.widgets.Button(
        name="\U0001f3e0", button_type="light", width=40,
        stylesheets=[_NAV_BTN_CSS])
    root_btn.on_click(lambda e: on_navigate("/"))
    items.append(root_btn)

    accumulated = ""
    for i, part in enumerate(parts):
        accumulated += f"/{part}"
        items.append(pn.pane.HTML(
            "<span style='color:#90a4ae; margin:0 2px; "
            "font-size:16px;'>\u203a</span>",
            margin=(6, 0)))
        is_last = (i == len(parts) - 1)
        btn = pn.widgets.Button(
            name=part,
            button_type="light" if not is_last else "default",
            stylesheets=[_NAV_BTN_CSS],
        )

        def _nav(event, _p=accumulated):
            on_navigate(_p)

        btn.on_click(_nav)
        items.append(btn)

    return pn.Row(
        *items,
        sizing_mode="stretch_width",
        stylesheets=[
            ":host { background: linear-gradient(135deg, #f5f7fa 0%, "
            "#e8ecf1 100%); border-radius: 6px; padding: 4px 10px; "
            "margin-bottom: 10px; }"
        ],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Group Browser View (main area)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_attribute_table(
    attributes: list[AttributeComparisonResult],
) -> pn.pane.Pane:
    """Build a table for attribute comparison."""
    if not attributes:
        return pn.pane.HTML(
            "<em style='color:#999;'>No differences in attributes</em>")

    df = pd.DataFrame([{
        "Attribute": a.name,
        "Reference Value": str(a.ref_value),
        "Target Value": str(a.tgt_value),
    } for a in attributes])

    return pn.widgets.Tabulator(
        df,
        show_index=False,
        sizing_mode="stretch_width",
        disabled=True,
        height=min(50 + len(df) * 35, 300),
        theme="bootstrap",
    )


def build_variable_table(
    variables: list[VariableComparisonResult],
    on_variable_click: Callable[[str], None],
) -> pn.widgets.Tabulator:
    """Build a Tabulator for variable comparison with on_click."""
    rows = []
    for v in variables:
        value_info = v.message or v.status.replace("_", " ").title()
        if v.stats is not None:
            value_info = f"{v.stats}, atol={v.atol}"
        short_name = v.varpath.rsplit("/", 1)[-1]
        rows.append({
            "Variable": short_name,
            "Status": v.status.replace("_", " ").title(),
            "Shape": "\u2260" if v.shape_diff else "\u2713",
            "DType": "\u2260" if v.dtype_diff else "\u2713",
            "Values": value_info,
            "_status": v.status,
            "_long_name": v.long_name,
            "_varpath": v.varpath,
        })

    df = pd.DataFrame(rows)

    # Row colour styling
    def _color_rows(row: pd.Series) -> list[str]:
        status = row.get("_status", "equal")
        bg = STATUS_COLORS.get(status, "#ffffff")
        fg = STATUS_TEXT_COLORS.get(status, "#000000")
        style = f"background-color: {bg}; color: {fg}"
        if status in ("different", "almost_equal"):
            style += "; cursor: pointer"
        return [style] * len(row)

    table = pn.widgets.Tabulator(
        df,
        show_index=False,
        sizing_mode="stretch_width",
        disabled=True,
        hidden_columns=["_status", "_long_name", "_varpath"],
        height=min(50 + len(df) * 35, 700),
        theme="bootstrap",
        selectable=1,
    )
    table.style.apply(_color_rows, axis=1)

    # Use both on_click AND selection watch for robustness
    def _on_cell_click(event) -> None:
        row_idx = event.row
        try:
            status = df.iloc[row_idx]["_status"]
            if status in ("different", "almost_equal"):
                varpath = df.iloc[row_idx]["_varpath"]
                on_variable_click(varpath)
        except (IndexError, KeyError):
            LOGGER.warning("Click on row %s out of range", row_idx)

    def _on_selection(event) -> None:
        if not event.new:
            return
        row_idx = event.new[0]
        try:
            status = df.iloc[row_idx]["_status"]
            if status in ("different", "almost_equal"):
                varpath = df.iloc[row_idx]["_varpath"]
                on_variable_click(varpath)
        except (IndexError, KeyError):
            LOGGER.warning("Selection row %s out of range", row_idx)

    table.on_click(_on_cell_click)
    table.param.watch(_on_selection, 'selection')

    return table


def build_group_browser(
    state: SessionState,
    group_path: str,
    on_group_click: Callable[[str], None],
    on_variable_click: Callable[[str], None],
) -> pn.Column:
    """Build the main-area view for a group."""
    gr = _find_group_result(state, group_path)
    content = pn.Column(sizing_mode="stretch_width", margin=(0, 10))

    if gr is None:
        content.append(pn.pane.Alert(
            f"Group `{group_path}` not found in comparison results.",
            alert_type="warning"))
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
            f"<span style='color:#2e7d32'>\u2713 {n_eq} equal</span>")
    if n_aeq:
        summary_parts.append(
            f"<span style='color:#006400'>\u2248 {n_aeq} almost equal</span>")
    if n_diff:
        summary_parts.append(
            f"<span style='color:#e65100'>\u2260 {n_diff} different</span>")
    if n_masked:
        summary_parts.append(
            f"<span style='color:#757575'>\u2014 {n_masked} masked</span>")

    if summary_parts:
        content.append(pn.pane.HTML(
            f"<div style='font-size:13px; padding:8px 12px; "
            f"background:#f5f7fa; border-radius:6px; margin-bottom:10px;'>"
            f"<b>{len(gr.variables)}</b> variables: "
            f"{' &nbsp;&bull;&nbsp; '.join(summary_parts)}"
            f"</div>",
            sizing_mode="stretch_width"))

    # Filter info
    if gr.filter_description:
        content.append(pn.pane.Alert(
            f"\U0001f50d {gr.filter_description}", alert_type="info"))

    # Sub-groups as styled buttons
    ref_group = _navigate_to_group(state.ref_ds, group_path)
    if ref_group is not None and ref_group.groups:
        content.append(pn.pane.HTML(
            "<h4 style='color:#37474f; margin:16px 0 8px 0;'>"
            "\U0001f4c2 Sub-groups</h4>"))
        btn_row: list[pn.viewable.Viewable] = []
        for grp_name in sorted(ref_group.groups):
            child_path = (f"{group_path}/{grp_name}"
                          if group_path != "/" else f"/{grp_name}")
            n_d, _n_t = _count_diffs_recursive(
                state, state.ref_ds, child_path)
            label = f"\U0001f4c1 {grp_name}"
            if n_d > 0:
                label += f" ({n_d} diffs)"
            btn_type = "warning" if n_d > 0 else "light"
            btn = pn.widgets.Button(
                name=label, button_type=btn_type,
                margin=(4, 6),
                stylesheets=[_NAV_BTN_CSS],
            )

            def _go(event, _p=child_path):
                on_group_click(_p)

            btn.on_click(_go)
            btn_row.append(btn)
        content.append(pn.FlexBox(*btn_row, sizing_mode="stretch_width"))

    # Attributes
    if gr.attributes:
        content.append(pn.pane.HTML(
            "<h4 style='color:#37474f; margin:16px 0 8px 0;'>"
            "\U0001f4cb Attributes</h4>"))
        content.append(build_attribute_table(gr.attributes))

    # Variables
    if gr.variables:
        content.append(pn.pane.HTML(
            "<h4 style='color:#37474f; margin:16px 0 8px 0;'>"
            "\U0001f4ca Variables</h4>"))
        if n_diff > 0:
            content.append(pn.pane.HTML(
                "<div style='font-size:11px; color:#78909c; "
                "margin-bottom:6px;'>"
                "Click a coloured row to see detailed comparison plots"
                "</div>"))
        content.append(build_variable_table(gr.variables, on_variable_click))

    return content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Detail View (main area)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_detail_view(
    state: SessionState,
    varpath: str,
) -> pn.Column:
    """Build the main-area detail view for a variable."""
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

    # Build the Bokeh dashboard
    figures_grid, plot_configs = build_dashboard(long_name, data_source)

    # CRITICAL: detach figures from the Bokeh grid layout so that
    # each can be independently rendered in its own Panel pane.
    figures_grid.children = []

    [config[0] for config in plot_configs]
    figures = [config[1] for config in plot_configs]

    # Fix figure sizing: override stretch_both → stretch_width so that
    # the figure's own height attribute is respected.
    for fig in figures:
        if hasattr(fig, 'sizing_mode'):
            fig.sizing_mode = "stretch_width"

    # All plots displayed flat — the only layout that renders
    # Bokeh figures reliably in Panel (no intermediate containers).
    plot_items: list[pn.viewable.Viewable] = []
    for label, fig in plot_configs:
        plot_items.append(pn.pane.HTML(
            f"<h4 style='color:#455a64; margin:16px 0 4px 0; "
            f"border-bottom:1px solid #e0e0e0; "
            f"padding-bottom:4px;'>{label}</h4>"))
        plot_items.append(pn.pane.Bokeh(
            fig, sizing_mode="stretch_width",
            margin=(0, 0, 20, 0)))

    # Marker size slider
    marker_slider = pn.widgets.IntSlider(
        name="Marker Size", start=1, end=10, value=2, step=1, width=200)

    def _on_marker(event) -> None:
        for fig in figures:
            if hasattr(fig, 'renderers'):
                for renderer in fig.renderers:
                    if hasattr(renderer, 'glyph') and hasattr(
                            renderer.glyph, 'size'):
                        renderer.glyph.size = event.new

    marker_slider.param.watch(_on_marker, 'value')

    # Flag filter
    flag_widget = None
    if data_source.flags:
        flag_widget = build_flag_filter(data_source)

    # Header
    status_bg = STATUS_COLORS.get(var_status, "#e0e0e0")
    status_fg = STATUS_TEXT_COLORS.get(var_status, "#000")
    status_label = var_status.replace("_", " ").title()
    header_html = (
        f"<div style='margin-bottom:10px;'>"
        f"<h3 style='margin:0 0 4px 0; color:#263238;'>"
        f"\U0001f4ca {varpath}</h3>"
        f"<span style='font-size:12px; color:#607d8b;'>{long_name}</span>"
        f" &nbsp; <span style='display:inline-block; padding:2px 10px; "
        f"border-radius:12px; font-size:11px; font-weight:600; "
        f"background:{status_bg}; color:{status_fg};'>"
        f"{status_label}</span>"
        f"</div>"
    )

    # Controls
    controls = pn.Row(
        pn.pane.HTML(
            "<span style='font-size:12px; color:#607d8b;'>"
            "<b>Marker size:</b></span>",
            margin=(8, 8, 0, 0)),
        marker_slider,
        sizing_mode="stretch_width",
        stylesheets=[_CARD_CSS],
    )

    detail = pn.Column(
        pn.pane.HTML(header_html, sizing_mode="stretch_width"),
        controls,
        sizing_mode="stretch_width",
        margin=(0, 10),
    )

    if flag_widget is not None:
        detail.append(flag_widget)

    detail.extend(plot_items)

    return detail


def build_flag_filter(data_source: DataSource) -> pn.Column:
    """Build flag filter widgets with Python callbacks."""
    all_options: list[str] = []
    for flag in data_source.flags:
        all_options.extend(flag.options)

    multi_choice = pn.widgets.MultiChoice(
        name="Flag Values",
        value=list(all_options),
        options=list(all_options),
        sizing_mode="stretch_width",
    )

    filter_input = pn.widgets.TextInput(
        name="Filter flags",
        placeholder="Filter flags (e.g., brown, *brown*)",
        sizing_mode="stretch_width",
    )

    def _on_filter(event) -> None:
        pattern = event.new.strip().lower()
        if not pattern:
            multi_choice.options = list(all_options)
            multi_choice.value = list(all_options)
            return
        filtered = [
            opt for opt in all_options
            if fnmatch.fnmatch(opt.lower(), f"*{pattern}*")
            or pattern in opt.lower()
        ]
        multi_choice.options = filtered
        multi_choice.value = filtered

    def _on_flag_change(event) -> None:
        selected = event.new
        if not selected:
            return
        indices = data_source.flags[0].compute_filtered_indices(
            selected, data_source.flags)
        data_source.apply_flag_filter(indices)

    filter_input.param.watch(_on_filter, 'value')
    multi_choice.param.watch(_on_flag_change, 'value')

    return pn.Column(
        pn.pane.HTML(
            "<b style='color:#37474f;'>\U0001f3f3 Flag Filters</b>"),
        filter_input, multi_choice,
        sizing_mode="stretch_width",
        stylesheets=[_CARD_CSS],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Expression Panel (sidebar)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_expression_panel(
    state: SessionState,
    refresh_callback: Callable[[], None],
) -> pn.Column:
    """Build the expression filter panel for the sidebar."""
    available_vars = list_all_variables(state.ref_ds)
    var_options = [f"${{{v}}}" for v in available_vars]

    # Variable selector — a Select dropdown is easier to browse than
    # autocomplete for large variable lists
    var_select = pn.widgets.Select(
        name="Insert variable",
        options=["", *var_options],
        value="",
        sizing_mode="stretch_width",
    )

    expr_area = pn.widgets.TextAreaInput(
        name="Expression",
        placeholder=(
            "Example: ${/data_20/distance_to_coast} > 30000"
        ),
        sizing_mode="stretch_width",
        height=80,
        value=state.active_expression or "",
    )

    status_pane = pn.pane.Alert("No filter active", alert_type="info")

    # Syntax help — collapsible
    syntax_help = pn.pane.HTML(
        "<div style='font-size:11px; color:#546e7a; line-height:1.5;'>"
        "<b>Syntax:</b> use <code>${/group/variable}</code> to reference "
        "a NetCDF variable.<br>"
        "<b>Operators:</b> <code>==</code> <code>!=</code> <code>&lt;</code>"
        " <code>&gt;</code> <code>&lt;=</code> <code>&gt;=</code>"
        " <code>&amp;</code> <code>|</code><br>"
        "<b>Functions:</b> <code>np.isin(var, [v1, v2])</code>, "
        "<code>np.abs(var)</code><br>"
        "<b>Combine:</b> <code>(expr1) &amp; (expr2)</code>"
        "</div>",
        sizing_mode="stretch_width",
    )

    # Example expressions as clickable buttons
    examples = [
        ("distance > 30km",
         "${/data_20/distance_to_coast} > 30000"),
        ("flag == 0",
         "${/data_01/surface_classification_flag} == 0"),
    ]
    example_btns: list[pn.viewable.Viewable] = []
    for ex_label, ex_expr in examples:
        btn = pn.widgets.Button(
            name=ex_label, button_type="light", width=140,
            stylesheets=[
                ":host .bk-btn { font-size:10px; padding:2px 6px; "
                "border:1px solid #cfd8dc; border-radius:12px; }"
            ],
        )

        def _insert_example(event, _expr=ex_expr):
            expr_area.value = _expr

        btn.on_click(_insert_example)
        example_btns.append(btn)

    def _on_var_select(event) -> None:
        picked = event.new
        if picked:
            current = expr_area.value
            if current and not current.endswith(
                    (" ", "(", "&", "|", "\n")):
                current += " "
            expr_area.value = current + picked
            var_select.value = ""

    def _apply(event) -> None:
        raw = expr_area.value.strip()
        if not raw:
            _clear(None)
            return
        rewritten = rewrite_expression(raw)
        mask, error = create_expression_mask(state.ref_ds, rewritten)
        if mask is None:
            status_pane.object = f"Error: {error}"
            status_pane.alert_type = "danger"
            return
        affected = find_affected_groups(raw)
        state.active_mask = mask
        state.active_expression = raw
        state.clear_variable_cache()
        selected = int(mask.sum())
        total = int(mask.size)
        masked = total - selected
        state.filter_info = FilterInfo(
            expression=raw,
            selected_points=selected,
            total_points=total,
            affected_groups=affected,
        )
        status_pane.object = (
            f"Filter active: {masked:,}/{total:,} points masked "
            f"({100 * masked / total:.1f}%)")
        status_pane.alert_type = "success"
        state.group_results = scan_all_groups(
            state.ref_ds, state.tgt_ds, state.filter_info,
            mask=state.active_mask)
        refresh_callback()

    def _clear(event) -> None:
        expr_area.value = ""
        state.active_mask = None
        state.active_expression = None
        state.filter_info = None
        state.clear_variable_cache()
        status_pane.object = "No filter active"
        status_pane.alert_type = "info"
        state.group_results = scan_all_groups(state.ref_ds, state.tgt_ds)
        refresh_callback()

    var_select.param.watch(_on_var_select, 'value')

    apply_btn = pn.widgets.Button(
        name="\u2713 Apply", button_type="primary", width=110)
    clear_btn = pn.widgets.Button(
        name="\u2715 Clear", button_type="warning", width=110)
    apply_btn.on_click(_apply)
    clear_btn.on_click(_clear)

    return pn.Column(
        pn.pane.HTML(
            "<div style='color:#37474f; font-weight:600; font-size:13px; "
            "letter-spacing:0.5px; text-transform:uppercase;'>"
            "\U0001f50d Expression Filter</div>"),
        syntax_help,
        pn.pane.HTML(
            "<div style='font-size:11px; color:#78909c; "
            "margin:4px 0 2px 0;'><b>Examples:</b></div>"),
        pn.Row(*example_btns),
        var_select,
        expr_area,
        pn.Row(apply_btn, clear_btn),
        status_pane,
        sizing_mode="stretch_width",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Legend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_legend() -> pn.pane.Pane:
    """Build a colour legend for the comparison table."""
    rows_html = ""
    for description, status in LEGEND_DATA:
        bg = STATUS_COLORS[status]
        rows_html += (
            f"<div style='display:flex; align-items:center; "
            f"margin-bottom:4px;'>"
            f"<span style='display:inline-block; width:16px; height:16px; "
            f"border-radius:3px; background-color:{bg}; "
            f"border:1px solid #ccc; margin-right:8px; flex-shrink:0;'>"
            f"</span>"
            f"<span style='font-size:11px; color:#333;'>"
            f"{description}</span></div>")

    html = (
        f"<div style='padding:8px 0;'>"
        f"<div style='color:#37474f; font-weight:600; font-size:13px; "
        f"letter-spacing:0.5px; text-transform:uppercase; "
        f"margin-bottom:8px;'>Legend</div>"
        f"{rows_html}</div>"
    )
    return pn.pane.HTML(html, sizing_mode="stretch_width")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HTML Export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_export_button(
    state: SessionState,
    main_container: pn.Column,
) -> pn.widgets.FileDownload:
    """Build an HTML export button."""

    def _generate_export() -> io.BytesIO:
        if state.current_varpath is None:
            buf = io.BytesIO()
            placeholder = pn.pane.Markdown(
                "# No variable selected\n\n"
                "Select a variable with differences to export.")
            placeholder.save(buf)
            buf.seek(0)
            return buf

        detail = build_detail_view(state, state.current_varpath)
        buf = io.BytesIO()
        try:
            detail.save(
                buf,
                title=f"Comparison: {state.current_varpath}",
                resources="cdn",
            )
        except Exception:
            LOGGER.exception("Export with resources=cdn failed, "
                             "trying inline")
            buf = io.BytesIO()
            detail.save(buf)
        buf.seek(0)
        return buf

    return pn.widgets.FileDownload(
        callback=_generate_export,
        filename="comparison_export.html",
        button_type="success",
        label="\U0001f4e5 Export HTML",
        sizing_mode="stretch_width",
    )
