"""Small editable table built from plain Dash inputs.

Replaces ``dash_table.DataTable`` (deprecated) in the annotation editor.
Each cell is a ``dcc.Input`` or ``dcc.Dropdown`` with the pattern id
``{"type": "<table>-cell", "row": i, "col": key}``; callbacks read a
table with :func:`cell_states` and rebuild rows with :func:`collect_rows`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

from dash import (
    ALL,
    MATCH,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    html,
    no_update,
)

_CELL = {
    "width": "100%",
    "padding": "4px 6px",
    "border": "1px solid #cfd8dc",
    "borderRadius": "4px",
    "fontSize": "12px",
    "fontFamily": "monospace",
    "boxSizing": "border-box",
}
_DEL = {
    "border": "none",
    "background": "none",
    "color": "#90a4ae",
    "cursor": "pointer",
    "fontSize": "14px",
    "padding": "0",
}


@dataclasses.dataclass(frozen=True)
class Column:
    """One column of a row editor.

    Attributes:
        key: Key of the value in the row dicts.
        label: Header text.
        kind: ``"text"``, ``"color"`` (text + swatch) or ``"multi"``
            (multi-select dropdown over ``options``).
        options: Choices of a ``"multi"`` column.
        width: CSS grid track size.
        placeholder: Placeholder of empty text cells.

    """

    key: str
    label: str
    kind: str = "text"
    options: tuple[str, ...] = ()
    width: str = "1fr"
    placeholder: str = ""


def _cell_id(table_id: str, row: int, key: str) -> dict[str, Any]:
    return {"type": f"{table_id}-cell", "row": row, "col": key}


def _grid(columns: list[Column], editable: bool) -> dict[str, str]:
    tracks = " ".join(c.width for c in columns) + (" 24px" if editable else "")
    return {
        "display": "grid",
        "gridTemplateColumns": tracks,
        "gap": "4px",
        "alignItems": "center",
        "marginBottom": "4px",
    }


def _cell(table_id: str, row: int, col: Column, value: Any, editable: bool) -> Any:
    cid = _cell_id(table_id, row, col.key)
    if col.kind == "multi":
        return dcc.Dropdown(
            id=cid,
            options=[{"label": o, "value": o} for o in col.options],
            value=list(value or []),
            multi=True,
            disabled=not editable,
            style={"fontSize": "12px"},
        )
    text = dcc.Input(
        id=cid,
        value="" if value is None else str(value),
        placeholder=col.placeholder,
        disabled=not editable,
        debounce=False,
        style=_CELL,
    )
    if col.kind != "color":
        return text
    return html.Div(
        [
            html.Span(
                id={"type": f"{table_id}-swatch", "row": row},
                style=swatch_style(value),
            ),
            text,
        ],
        style={"display": "flex", "gap": "4px", "alignItems": "center"},
    )


def swatch_style(color: Any) -> dict[str, str]:
    """Style of the colour preview next to a ``"color"`` cell."""
    return {
        "width": "18px",
        "height": "18px",
        "flexShrink": "0",
        "borderRadius": "3px",
        "border": "1px solid #cfd8dc",
        "backgroundColor": str(color or "transparent"),
    }


def build_rows(
    table_id: str,
    columns: list[Column],
    rows: list[dict[str, Any]],
    *,
    editable: bool = True,
) -> list[Any]:
    """Build the row components of a table."""
    out = []
    for i, row in enumerate(rows):
        cells = [_cell(table_id, i, c, row.get(c.key), editable) for c in columns]
        if editable:
            cells.append(
                html.Button(
                    "✕",
                    id={"type": f"{table_id}-del", "row": i},
                    n_clicks=0,
                    title="Delete row",
                    style=_DEL,
                )
            )
        out.append(html.Div(cells, style=_grid(columns, editable)))
    return out


def build_row_editor(
    table_id: str,
    columns: list[Column],
    rows: list[dict[str, Any]],
    *,
    editable: bool = True,
    add_label: str = "+ Add row",
) -> html.Div:
    """Header, rows and *add* button of an editable table."""
    header = html.Div(
        [
            html.Span(
                c.label,
                style={"fontSize": "11px", "fontWeight": "600", "color": "#546e7a"},
            )
            for c in columns
        ]
        + ([html.Span()] if editable else []),
        style=_grid(columns, editable),
    )
    children: list[Any] = [
        header,
        html.Div(
            build_rows(table_id, columns, rows, editable=editable),
            id=f"{table_id}-rows",
        ),
    ]
    if editable:
        children.append(
            html.Button(
                add_label,
                id=f"{table_id}-add",
                n_clicks=0,
                style={
                    "border": "1px solid #cfd8dc",
                    "borderRadius": "4px",
                    "padding": "4px 10px",
                    "backgroundColor": "white",
                    "cursor": "pointer",
                    "fontSize": "12px",
                    "marginTop": "4px",
                },
            )
        )
    return html.Div(children, id=table_id)


def cell_states(table_id: str) -> list[State]:
    """States giving the values and ids of every cell of a table."""
    pattern = {"type": f"{table_id}-cell", "row": ALL, "col": ALL}
    return [State(pattern, "value"), State(pattern, "id")]


def collect_rows(
    values: list[Any] | None, ids: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Rebuild row dicts (in row order) from cell values and ids."""
    rows: dict[int, dict[str, Any]] = {}
    for value, cid in zip(values or [], ids or [], strict=False):
        rows.setdefault(int(cid["row"]), {})[cid["col"]] = value
    return [rows[i] for i in sorted(rows)]


def register_row_editor(
    app: Any,
    table_id: str,
    columns: list[Column],
    blank_row: Callable[..., dict[str, Any]],
    extra_states: list[State] | None = None,
) -> None:
    """Register the add/delete (and colour preview) callbacks of a table.

    ``blank_row`` receives the values of ``extra_states`` and returns the
    row appended by the *add* button.
    """
    extra_states = extra_states or []

    @app.callback(
        Output(f"{table_id}-rows", "children"),
        Input(f"{table_id}-add", "n_clicks"),
        Input({"type": f"{table_id}-del", "row": ALL}, "n_clicks"),
        *cell_states(table_id),
        *extra_states,
        prevent_initial_call=True,
    )
    def _edit_rows(_add, _dels, values, ids, *extra):
        trig = next(
            (t for t in callback_context.triggered if (t.get("value") or 0) > 0),
            None,
        )
        if trig is None:
            return no_update
        rows = collect_rows(values, ids)
        prop = trig["prop_id"].rsplit(".", 1)[0]
        if prop == f"{table_id}-add":
            rows.append(blank_row(*extra))
        else:
            del rows[int(json.loads(prop)["row"])]
        return build_rows(table_id, columns, rows)

    if any(c.kind == "color" for c in columns):
        color = next(c.key for c in columns if c.kind == "color")
        app.clientside_callback(
            """
            function(value, style) {
                var s = Object.assign({}, style || {});
                s.backgroundColor = value || 'transparent';
                return s;
            }
            """,
            Output({"type": f"{table_id}-swatch", "row": MATCH}, "style"),
            Input({"type": f"{table_id}-cell", "row": MATCH, "col": color}, "value"),
            State({"type": f"{table_id}-swatch", "row": MATCH}, "style"),
            prevent_initial_call=True,
        )
