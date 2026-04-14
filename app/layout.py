"""Dash layout components for the NetCDF comparison application."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dash import dcc, html

if TYPE_CHECKING:
    from .state import SessionState

from .constants import LEGEND_DATA, STATUS_COLORS
from .expression import list_all_variables


def build_legend_html() -> html.Div:
    """Build the colour legend."""
    items = []
    for description, status in LEGEND_DATA:
        bg = STATUS_COLORS[status]
        items.append(
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "16px",
                            "height": "16px",
                            "borderRadius": "3px",
                            "backgroundColor": bg,
                            "border": "1px solid #ccc",
                            "marginRight": "8px",
                            "flexShrink": "0",
                        }
                    ),
                    html.Span(
                        description,
                        style={
                            "fontSize": "11px",
                            "color": "#333",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "4px",
                },
            )
        )

    return html.Div(
        [
            html.Div(
                "Legend",
                style={
                    "color": "#37474f",
                    "fontWeight": "600",
                    "fontSize": "13px",
                    "letterSpacing": "0.5px",
                    "textTransform": "uppercase",
                    "marginBottom": "8px",
                },
            ),
            *items,
        ],
        style={"padding": "8px 0"},
    )


def build_file_info(state: SessionState) -> html.Div:
    """Build file info cards."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "Source",
                        style={
                            "fontSize": "11px",
                            "color": "#1565c0",
                            "textTransform": "uppercase",
                            "fontWeight": "600",
                            "letterSpacing": "0.5px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        state.src_path.name,
                        style={
                            "fontSize": "12px",
                            "color": "#263238",
                            "wordBreak": "break-all",
                        },
                    ),
                ],
                style={
                    "background": "linear-gradient(135deg, #e3f2fd, #bbdefb)",
                    "borderRadius": "8px",
                    "padding": "10px 14px",
                    "marginBottom": "4px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        "Target",
                        style={
                            "fontSize": "11px",
                            "color": "#c62828",
                            "textTransform": "uppercase",
                            "fontWeight": "600",
                            "letterSpacing": "0.5px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        state.tgt_path.name,
                        style={
                            "fontSize": "12px",
                            "color": "#263238",
                            "wordBreak": "break-all",
                        },
                    ),
                ],
                style={
                    "background": "linear-gradient(135deg, #fce4ec, #f8bbd0)",
                    "borderRadius": "8px",
                    "padding": "10px 14px",
                },
            ),
        ]
    )


def _btn_style(bg: str = "#1565c0") -> dict:
    """Return a standard small button style."""
    return {
        "backgroundColor": bg,
        "color": "white",
        "border": "none",
        "borderRadius": "4px",
        "padding": "6px 16px",
        "cursor": "pointer",
        "fontSize": "12px",
    }


def build_expression_panel(state: SessionState) -> html.Div:
    """Build the visual filter-builder panel for the sidebar."""
    available_vars = list_all_variables(state.src_ds)
    var_options = [{"label": v, "value": v} for v in available_vars]

    operator_options = [
        {"label": ">", "value": ">"},
        {"label": "<", "value": "<"},
        {"label": ">=", "value": ">="},
        {"label": "<=", "value": "<="},
        {"label": "==", "value": "=="},
        {"label": "!=", "value": "!="},
        {"label": "in", "value": "in"},
    ]

    _row = {
        "display": "flex",
        "gap": "6px",
        "marginBottom": "6px",
        "alignItems": "center",
    }
    _label = {
        "fontSize": "11px",
        "color": "#607d8b",
        "fontWeight": "600",
        "minWidth": "55px",
    }

    return html.Div(
        [
            html.Div(
                "Data Filter",
                style={
                    "color": "#37474f",
                    "fontWeight": "600",
                    "fontSize": "13px",
                    "letterSpacing": "0.5px",
                    "textTransform": "uppercase",
                    "marginBottom": "10px",
                },
            ),
            # ── Store: list of conditions [{var, op, val}, ...] ────────
            dcc.Store(id="filter-conditions", data=[]),
            # ── Condition builder ──────────────────────────────────────
            html.Div(
                [
                    # Row 1: variable
                    html.Div(
                        [
                            html.Span("Variable", style=_label),
                            dcc.Dropdown(
                                id="filter-var",
                                options=var_options,
                                placeholder="Select...",
                                style={"flex": "1", "fontSize": "11px"},
                            ),
                        ],
                        style=_row,
                    ),
                    # Row 2: operator
                    html.Div(
                        [
                            html.Span("Operator", style=_label),
                            dcc.Dropdown(
                                id="filter-op",
                                options=operator_options,
                                value=">",
                                clearable=False,
                                style={"width": "80px", "fontSize": "11px"},
                            ),
                        ],
                        style=_row,
                    ),
                    # Row 3: value
                    html.Div(
                        [
                            html.Span("Value", style=_label),
                            dcc.Input(
                                id="filter-val",
                                type="text",
                                placeholder="e.g. 30000 or [0,1,2]",
                                style={
                                    "flex": "1",
                                    "fontSize": "11px",
                                    "padding": "4px 8px",
                                    "border": "1px solid #ccc",
                                    "borderRadius": "4px",
                                },
                            ),
                        ],
                        style=_row,
                    ),
                    # Variable range hint
                    html.Div(
                        id="filter-var-hint",
                        style={
                            "fontSize": "10px",
                            "color": "#90a4ae",
                            "marginBottom": "6px",
                            "fontStyle": "italic",
                        },
                    ),
                    # Add button
                    html.Button(
                        "+ Add condition",
                        id="filter-add-btn",
                        n_clicks=0,
                        style=_btn_style("#43a047"),
                    ),
                ],
                style={
                    "padding": "10px",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "6px",
                    "backgroundColor": "#fff",
                    "marginBottom": "10px",
                },
            ),
            # ── Active conditions list ─────────────────────────────────
            html.Div(
                id="filter-conditions-display",
                style={
                    "marginBottom": "10px",
                },
            ),
            # ── Preview ────────────────────────────────────────────────
            html.Div(
                id="filter-preview",
                style={
                    "fontSize": "12px",
                    "padding": "8px 12px",
                    "borderRadius": "4px",
                    "backgroundColor": "#f5f7fa",
                    "color": "#607d8b",
                    "marginBottom": "10px",
                },
            ),
            # ── Apply / Clear ──────────────────────────────────────────
            html.Div(
                [
                    html.Button(
                        "Apply",
                        id="filter-apply-btn",
                        n_clicks=0,
                        style=_btn_style("#1565c0"),
                    ),
                    html.Button(
                        "Clear all",
                        id="filter-clear-btn",
                        n_clicks=0,
                        style={**_btn_style("#ff9800"), "marginLeft": "8px"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            # ── Status ─────────────────────────────────────────────────
            html.Div(
                id="expr-status",
                children="No filter active",
                style={
                    "fontSize": "12px",
                    "padding": "8px 12px",
                    "borderRadius": "4px",
                    "backgroundColor": "#e3f2fd",
                    "color": "#1565c0",
                },
            ),
            # ── Advanced mode toggle ───────────────────────────────────
            html.Details(
                [
                    html.Summary(
                        "Advanced mode",
                        style={
                            "fontSize": "11px",
                            "color": "#90a4ae",
                            "cursor": "pointer",
                            "marginBottom": "6px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.B("Syntax: "),
                                    html.Code("${/group/variable}"),
                                    " references a NetCDF variable.",
                                ],
                                style={"marginBottom": "4px"},
                            ),
                            html.Div(
                                [
                                    html.B("Combine: "),
                                    html.Code("(expr1) & (expr2)"),
                                    " or ",
                                    html.Code("(expr1) | (expr2)"),
                                ],
                                style={"marginBottom": "4px"},
                            ),
                            html.Div(
                                [
                                    html.B("Functions: "),
                                    html.Code("np.isin(var, [v1, v2])"),
                                    ", ",
                                    html.Code("np.abs(var)"),
                                ],
                                style={"marginBottom": "6px"},
                            ),
                        ],
                        style={
                            "fontSize": "11px",
                            "color": "#546e7a",
                            "lineHeight": "1.5",
                            "marginBottom": "6px",
                        },
                    ),
                    dcc.Textarea(
                        id="expr-textarea",
                        placeholder=(
                            "e.g. (${/data_20/distance_to_coast} > 30000) "
                            "& (${/data_01/surface_classification_flag} == 0)"
                        ),
                        style={
                            "width": "100%",
                            "height": "70px",
                            "fontSize": "11px",
                            "fontFamily": "monospace",
                        },
                        value=state.active_expression or "",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Apply expression",
                                id="expr-apply-btn",
                                n_clicks=0,
                                style={**_btn_style("#1565c0"), "marginTop": "6px"},
                            ),
                        ]
                    ),
                ],
                style={"marginTop": "10px"},
            ),
        ]
    )


def create_layout(state: SessionState) -> html.Div:
    """Build the full Dash layout.

    Args:
        state: The session state.

    Returns:
        Root Dash component.

    """
    return html.Div(
        [
            # ── Hidden stores ──────────────────────────────────────────
            dcc.Store(id="nav-current-view", data="group"),
            dcc.Store(id="nav-group-path", data="/"),
            dcc.Store(id="nav-varpath", data=None),
            dcc.Store(id="zoom-range", data=None),
            dcc.Store(id="url-initial-load", data=True),
            dcc.Location(id="url-location", refresh=False),
            # ── Header ─────────────────────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "\u2630",
                                id="sidebar-toggle-btn",
                                n_clicks=0,
                                style={
                                    "background": "none",
                                    "border": "1px solid rgba(255,255,255,0.5)",
                                    "borderRadius": "4px",
                                    "color": "white",
                                    "fontSize": "18px",
                                    "cursor": "pointer",
                                    "padding": "2px 8px",
                                    "marginRight": "12px",
                                    "lineHeight": "1",
                                },
                            ),
                            html.Span(
                                state.src_path.name,
                                style={
                                    "color": "white",
                                    "fontWeight": "500",
                                    "fontSize": "14px",
                                },
                            ),
                            html.Span(
                                " vs ",
                                style={
                                    "color": "rgba(255,255,255,0.6)",
                                    "fontSize": "13px",
                                    "margin": "0 6px",
                                },
                            ),
                            html.Span(
                                state.tgt_path.name,
                                style={
                                    "color": "white",
                                    "fontWeight": "500",
                                    "fontSize": "14px",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        [
                            html.Div(id="filter-banner", style={"display": "none"}),
                            html.Button(
                                "\U0001f517 Share",
                                id="share-btn",
                                n_clicks=0,
                                style={
                                    "border": "1px solid rgba(255,255,255,0.5)",
                                    "borderRadius": "4px",
                                    "padding": "4px 12px",
                                    "backgroundColor": "transparent",
                                    "color": "white",
                                    "cursor": "pointer",
                                    "fontSize": "12px",
                                    "marginLeft": "8px",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                            html.Div(id="share-toast", style={"display": "none"}),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    ),
                ],
                style={
                    "backgroundColor": "#1565c0",
                    "padding": "12px 24px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                },
            ),
            # ── Body: sidebar + main ───────────────────────────────────
            html.Div(
                [
                    # Sidebar
                    html.Div(
                        [
                            build_file_info(state),
                            html.Hr(style={"margin": "12px 0"}),
                            build_legend_html(),
                            html.Button(
                                "Export HTML",
                                id="export-btn",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "backgroundColor": "#4caf50",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "padding": "8px",
                                    "cursor": "pointer",
                                    "fontSize": "13px",
                                    "marginBottom": "8px",
                                },
                            ),
                            dcc.Download(id="export-download"),
                            html.Hr(style={"margin": "12px 0"}),
                            build_expression_panel(state),
                        ],
                        id="sidebar",
                        style={
                            "width": "380px",
                            "minWidth": "380px",
                            "padding": "16px",
                            "backgroundColor": "#fafafa",
                            "borderRight": "1px solid #e0e0e0",
                            "overflowY": "auto",
                            "height": "calc(100vh - 52px)",
                            "transition": "margin-left 0.3s ease, opacity 0.3s ease",
                        },
                    ),
                    # Main content area
                    html.Div(
                        id="main-content",
                        style={
                            "flex": "1",
                            "padding": "16px 24px",
                            "overflowY": "auto",
                            "height": "calc(100vh - 52px)",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "height": "calc(100vh - 52px)",
                },
            ),
        ],
        style={"fontFamily": "'Segoe UI', Roboto, sans-serif"},
    )
