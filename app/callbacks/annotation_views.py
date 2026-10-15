"""Dash components displaying and editing the annotation document.

Pure builders (no callback registration): badges for the variable
tables, the annotation panel of the detail view, the annotation form
shared by the detail-view drawer and the editor page, and the editor
page itself (document, categories, outline).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dash import dcc, html

from ..annotations import (
    AUTO_SECTIONS,
    DIFFERENT_STATUSES,
    LANGUAGES,
    STATUSES,
    Annotation,
    AnnotationStore,
    FigureSpec,
    Section,
    match_annotations,
    pattern_matches,
)
from ..constants import PLOT_LABELS
from .row_editor import Column, build_row_editor

if TYPE_CHECKING:
    from ..annotations import AnnotationDocument

_ACCENT = "rgb(76,114,176)"
_MUTED = "#607d8b"

_BTN = {
    "border": "1px solid #cfd8dc",
    "borderRadius": "4px",
    "padding": "4px 10px",
    "backgroundColor": "white",
    "cursor": "pointer",
    "fontSize": "12px",
}
_BTN_PRIMARY = {**_BTN, "backgroundColor": _ACCENT, "color": "white", "border": "none"}
_BTN_DANGER = {**_BTN, "color": "#c62828", "borderColor": "#ef9a9a"}
_LABEL = {
    "fontSize": "11px",
    "color": _MUTED,
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.4px",
    "marginBottom": "2px",
    "display": "block",
}
_INPUT = {
    "width": "100%",
    "padding": "5px 8px",
    "border": "1px solid #cfd8dc",
    "borderRadius": "4px",
    "fontSize": "13px",
    "boxSizing": "border-box",
}
_TEXTAREA = {**_INPUT, "fontFamily": "monospace", "fontSize": "12px"}
_CARD = {
    "border": "1px solid #e0e0e0",
    "borderRadius": "8px",
    "backgroundColor": "white",
    "padding": "10px 14px",
    "marginBottom": "10px",
}
_AUTO_LABELS = {
    "summary_counts": "Summary counts",
    "agreement_table": "Agreement table",
    "unannotated": "Unannotated differences",
}
#: Marks the button clicked by Ctrl+S (see assets/ncjuxta-annotations.js).
_CTRL_S: dict[str, Any] = {"data-ctrl-s": "1"}
#: Columns of the editable tables (see :mod:`.row_editor`).
FIGURE_COLUMNS = [
    Column("variable", "Variable", width="1.3fr"),
    Column("plots", "Plots", kind="multi", options=PLOT_LABELS, width="1.5fr"),
    Column("expression", "Filter expression", width="1.3fr"),
    Column("zoom", "Zoom (min, max)", width="0.8fr", placeholder="e.g. 100, 250"),
    Column("caption", "Caption (Markdown)", width="1.3fr"),
]
DATASET_COLUMNS = [
    Column("label", "Label", width="1fr"),
    Column("value", "Value", width="3fr"),
]
CATEGORY_COLUMNS = [
    Column("id", "Id", width="1fr"),
    Column("label", "Label", width="1.5fr"),
    Column("color", "Colour (#rrggbb)", kind="color", width="1fr"),
    Column("description", "Description", width="2.5fr"),
]
_STATUS_LABELS = {
    "open": "Open",
    "confirmed": "Confirmed",
    "resolved": "Resolved",
    "wontfix": "Won't fix",
}


def get_store(state: Any) -> AnnotationStore | None:
    """Return the annotation store of the app, if one was configured."""
    store = getattr(state, "annotations", None)
    return store if isinstance(store, AnnotationStore) else None


def is_editable(state: Any) -> bool:
    """Return True if annotations are loaded and may be edited."""
    store = get_store(state)
    return store is not None and not store.readonly


def all_varpaths(state: Any) -> list[str]:
    """Every variable path of the current comparison results."""
    return [v.varpath for gr in state.group_results for v in gr.variables]


def coverage(state: Any) -> dict[str, list[Annotation]]:
    """Map variable paths to the annotations covering them."""
    store = get_store(state)
    if store is None:
        return {}
    return match_annotations(store.document, all_varpaths(state))


def _category(doc: AnnotationDocument, category_id: str) -> tuple[str, str]:
    cat = doc.category(category_id)
    if cat is None:
        return "Uncategorised", "#9e9e9e"
    return cat.label or cat.id, cat.color


def category_badge(doc: AnnotationDocument, category_id: str) -> html.Span:
    """Coloured pill with the category label."""
    label, color = _category(doc, category_id)
    return html.Span(
        label,
        style={
            "backgroundColor": color,
            "color": "white",
            "borderRadius": "3px",
            "padding": "1px 8px",
            "fontSize": "11px",
            "fontWeight": "600",
            "whiteSpace": "nowrap",
        },
    )


def status_badge(status: str) -> html.Span:
    """Outlined pill with the annotation status."""
    return html.Span(
        _STATUS_LABELS.get(status, status),
        style={
            "border": "1px solid #cfd8dc",
            "borderRadius": "3px",
            "padding": "0 6px",
            "fontSize": "11px",
            "color": _MUTED,
            "whiteSpace": "nowrap",
        },
    )


def annotation_dots(doc: AnnotationDocument, annotations: list[Annotation]) -> Any:
    """Compact markers (one dot per annotation) for table cells."""
    if not annotations:
        return ""
    return html.Span(
        [
            html.Span(
                "●",
                title=f"{_category(doc, a.category)[0]} — {a.title or a.id}",
                style={
                    "color": _category(doc, a.category)[1],
                    "fontSize": "14px",
                    "marginRight": "2px",
                },
            )
            for a in annotations
        ]
        + [
            html.Span(
                annotations[0].title or annotations[0].id,
                style={
                    "fontSize": "11px",
                    "marginLeft": "4px",
                    "maxWidth": "240px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                    "display": "inline-block",
                    "verticalAlign": "middle",
                },
            )
        ],
        style={"whiteSpace": "nowrap"},
    )


def build_annotation_overview(state: Any) -> html.Div | None:
    """Root-page summary: annotations per category and coverage."""
    store = get_store(state)
    if store is None:
        return None
    doc = store.document
    cov = coverage(state)
    differing = [
        v.varpath
        for gr in state.group_results
        for v in gr.variables
        if v.status in DIFFERENT_STATUSES
    ]
    unannotated = [p for p in differing if p not in cov]
    counts: dict[str, int] = {}
    for ann in doc.annotations:
        counts[ann.category] = counts.get(ann.category, 0) + 1
    chips: list[Any] = []
    for cat in doc.categories:
        if counts.get(cat.id):
            chips.append(
                html.Span(
                    [category_badge(doc, cat.id), f" {counts[cat.id]}"],
                    style={"marginRight": "12px", "fontSize": "12px"},
                )
            )
    if counts.get(""):
        chips.append(
            html.Span(
                [category_badge(doc, ""), f" {counts['']}"],
                style={"marginRight": "12px", "fontSize": "12px"},
            )
        )
    cover_color = "#c62828" if unannotated else "#2e7d32"
    return html.Div(
        [
            html.Span(
                "\U0001f4dd Annotations",
                style={"fontWeight": "600", "marginRight": "12px"},
            ),
            html.Span(
                f"{len(doc.annotations)} annotation(s)",
                style={"marginRight": "12px", "fontSize": "12px"},
            ),
            *chips,
            html.Span(
                f"{len(differing) - len(unannotated)}/{len(differing)} "
                "differing variables annotated",
                title="\n".join(unannotated[:50]),
                style={"fontSize": "12px", "color": cover_color, "fontWeight": "600"},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "flexWrap": "wrap",
            "gap": "4px",
            "padding": "8px 12px",
            "backgroundColor": "#fffde7",
            "border": "1px solid #fff59d",
            "borderRadius": "6px",
            "marginBottom": "12px",
        },
    )


# ── Detail view panel ───────────────────────────────────────────────


def _figure_summary(spec: FigureSpec) -> str:
    parts = [", ".join(spec.plots) if spec.plots else "all plots"]
    if spec.expression:
        parts.append(spec.expression)
    if spec.zoom is not None:
        parts.append(f"zoom [{spec.zoom[0]:g} ; {spec.zoom[1]:g}]")
    return " · ".join(parts)


def annotation_card(
    doc: AnnotationDocument,
    ann: Annotation,
    *,
    editable: bool,
    varpath: str | None = None,
) -> html.Div:
    """Read-only rendering of one annotation, with its action buttons."""
    facts: list[Any] = []
    for label, value in (
        ("Variables", ", ".join(ann.variables)),
        ("Magnitude", ann.magnitude),
        ("Cause", ann.cause),
        ("References", ", ".join(ann.refs)),
    ):
        if value:
            facts.append(
                html.Div(
                    [
                        html.Span(
                            label,
                            style={"color": _MUTED, "width": "90px", "flexShrink": 0},
                        ),
                        html.Span(value),
                    ],
                    style={"display": "flex", "gap": "8px", "fontSize": "12px"},
                )
            )
    body: list[Any] = [html.Div(facts, style={"marginBottom": "6px"})]
    if ann.text:
        body.append(dcc.Markdown(ann.text, style={"fontSize": "13px"}))
    if ann.effect:
        body.append(
            html.Div(
                [html.B("Effect: "), dcc.Markdown(ann.effect)],
                style={
                    "borderLeft": f"3px solid {_ACCENT}",
                    "padding": "2px 10px",
                    "backgroundColor": "#f5f7fa",
                    "fontSize": "13px",
                },
            )
        )
    figs: list[Any] = []
    for idx, spec in enumerate(ann.figures):
        if varpath is not None and spec.variable != varpath:
            continue
        figs.append(
            html.Div(
                [
                    html.Span(
                        _figure_summary(spec),
                        style={"fontSize": "12px", "color": _MUTED, "flex": "1"},
                    ),
                    html.Button(
                        "\U0001f441 Show this view",
                        id={"type": "ann-showfig", "ann": ann.id, "idx": idx},
                        n_clicks=0,
                        style=_BTN,
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px"},
            )
        )
    if figs:
        body.append(html.Div(figs, style={"marginTop": "6px"}))
    actions: list[Any] = []
    if editable:
        actions = [
            html.Button(
                "✎ Edit",
                id={"type": "ann-edit", "id": ann.id},
                n_clicks=0,
                style=_BTN,
            )
        ]
        if varpath is not None:
            actions.append(
                html.Button(
                    "+ Add current view as figure",
                    id={"type": "ann-addfig", "id": ann.id},
                    n_clicks=0,
                    style=_BTN,
                )
            )
    return html.Div(
        [
            html.Div(
                [
                    category_badge(doc, ann.category),
                    status_badge(ann.status),
                    html.Span(
                        ann.title or ann.id,
                        style={"fontWeight": "600", "flex": "1", "fontSize": "14px"},
                    ),
                    *actions,
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "marginBottom": "6px",
                    "flexWrap": "wrap",
                },
            ),
            *body,
        ],
        style=_CARD,
    )


def build_annotation_cards(state: Any, varpath: str) -> list[Any]:
    """Cards of the annotations covering ``varpath``."""
    store = get_store(state)
    if store is None:
        return []
    doc = store.document
    anns = [
        a
        for a in doc.annotations
        if any(pattern_matches(p, varpath) for p in a.variables)
    ]
    if not anns:
        return [
            html.Div(
                "No annotation covers this variable yet.",
                style={"fontSize": "12px", "color": _MUTED, "padding": "4px 0"},
            )
        ]
    editable = not store.readonly
    return [annotation_card(doc, a, editable=editable, varpath=varpath) for a in anns]


def build_annotation_panel(state: Any, varpath: str) -> html.Details | None:
    """Collapsible panel listing the annotations of a variable."""
    store = get_store(state)
    if store is None:
        return None
    n = len(coverage(state).get(varpath, []))
    header: list[Any] = [
        html.Span(
            f"\U0001f4dd Annotations ({n})",
            style={"fontWeight": "600", "fontSize": "13px", "color": "#37474f"},
        )
    ]
    if not store.readonly:
        header.append(
            html.Button(
                "+ Annotate this variable",
                id="ann-new-for-var",
                n_clicks=0,
                style={**_BTN_PRIMARY, "marginLeft": "auto"},
            )
        )
    return html.Details(
        [
            html.Summary(
                header,
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "cursor": "pointer",
                    "listStyle": "none",
                },
            ),
            html.Div(
                build_annotation_cards(state, varpath),
                id="annotation-panel-cards",
                style={"marginTop": "8px"},
            ),
            html.Div(id="ann-panel-msg", style={"fontSize": "12px"}),
            html.Div(id="ann-drawer"),
        ],
        open=n > 0,
        id="annotation-panel",
        style={
            "padding": "8px 14px",
            "border": "1px solid #fff59d",
            "borderRadius": "8px",
            "backgroundColor": "#fffde7",
            "marginBottom": "10px",
        },
    )


# ── Forms ───────────────────────────────────────────────────────────


def _field(label: str, component: Any, width: str = "100%") -> html.Div:
    return html.Div(
        [html.Label(label, style=_LABEL), component],
        style={"width": width, "marginBottom": "8px"},
    )


def _section_options(doc: AnnotationDocument) -> list[dict[str, str]]:
    return [
        {"label": "\u2003" * depth + (s.title or s.id), "value": s.id}
        for s, _, depth in doc.iter_sections()
    ]


def _section_of(doc: AnnotationDocument, annotation_id: str) -> str | None:
    for section, _, _ in doc.iter_sections():
        if annotation_id in section.annotations:
            return section.id
    return None


def figures_to_rows(figures: list[FigureSpec]) -> list[dict[Any, Any]]:
    """Convert figure specs to rows of the editable figure table."""
    return [
        {
            "variable": f.variable,
            "plots": list(f.plots),
            "expression": f.expression,
            "zoom": f"{f.zoom[0]:g}, {f.zoom[1]:g}" if f.zoom else "",
            "caption": f.caption,
        }
        for f in figures
    ]


def rows_to_figures(
    rows: list[dict[str, Any]] | None,
) -> tuple[list[FigureSpec], list[str]]:
    """Parse the figure table rows; return ``(figures, errors)``."""
    figures: list[FigureSpec] = []
    errors: list[str] = []
    for i, row in enumerate(rows or [], start=1):
        variable = str(row.get("variable") or "").strip()
        if not variable:
            continue
        raw_plots = row.get("plots") or []
        if isinstance(raw_plots, str):
            raw_plots = raw_plots.split(",")
        plots = [str(p).strip() for p in raw_plots if str(p).strip()]
        zoom_text = str(row.get("zoom") or "").strip()
        zoom = None
        if zoom_text:
            try:
                lo, hi = (float(v) for v in zoom_text.replace(";", ",").split(","))
                zoom = [lo, hi]
            except ValueError:
                errors.append(f"figure {i}: zoom must be 'min, max'")
        figures.append(
            FigureSpec(
                variable=variable,
                plots=plots,
                expression=str(row.get("expression") or "").strip(),
                zoom=zoom,
                caption=str(row.get("caption") or ""),
            )
        )
    return figures, errors


def build_annotation_form(
    store: AnnotationStore,
    annotation: Annotation | None,
    *,
    original_id: str | None,
    section_id: str | None = None,
    context: str = "editor",
) -> html.Div:
    """Edit form for one annotation.

    Args:
        store: The annotation store.
        annotation: Values to show (``None`` for an empty new annotation).
        original_id: Id of the annotation being edited, ``None`` when new.
        section_id: Section to attach the annotation to.
        context: ``"editor"`` or ``"drawer"`` (detail view), for layout.

    """
    doc = store.document
    ann = annotation or Annotation(id="")
    if section_id is None and original_id:
        section_id = _section_of(doc, original_id)
    cat_options = [{"label": c.label or c.id, "value": c.id} for c in doc.categories]
    title = "New annotation" if original_id is None else f"Edit “{original_id}”"
    figure_table = build_row_editor(
        "ann-form-figures",
        FIGURE_COLUMNS,
        figures_to_rows(ann.figures),
        add_label="+ Add figure",
    )
    row = {"display": "flex", "gap": "12px", "flexWrap": "wrap"}
    return html.Div(
        [
            dcc.Store(id="ann-form-original", data=original_id),
            dcc.Store(id="ann-form-revision", data=store.revision),
            html.Div(
                [
                    html.H4(title, style={"margin": "0", "flex": "1"}),
                    html.Span(
                        id="ann-form-dirty",
                        style={"fontSize": "12px", "color": "#e65100"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                [
                    _field(
                        "Title",
                        dcc.Input(id="ann-form-title", value=ann.title, style=_INPUT),
                        "60%",
                    ),
                    _field(
                        "Id",
                        dcc.Input(
                            id="ann-form-id",
                            value=ann.id,
                            placeholder="generated from the title",
                            style=_INPUT,
                        ),
                        "calc(40% - 12px)",
                    ),
                ],
                style=row,
            ),
            html.Div(
                [
                    _field(
                        "Category",
                        dcc.Dropdown(
                            id="ann-form-category",
                            options=cat_options,
                            value=ann.category or None,
                            placeholder="Uncategorised",
                        ),
                        "30%",
                    ),
                    _field(
                        "Status",
                        dcc.Dropdown(
                            id="ann-form-status",
                            options=[
                                {"label": _STATUS_LABELS[s], "value": s}
                                for s in STATUSES
                            ],
                            value=ann.status,
                            clearable=False,
                        ),
                        "20%",
                    ),
                    _field(
                        "Section",
                        dcc.Dropdown(
                            id="ann-form-section",
                            options=_section_options(doc),
                            value=section_id,
                            placeholder="Not placed (“Other annotations”)",
                        ),
                        "calc(50% - 24px)",
                    ),
                ],
                style=row,
            ),
            _field(
                "Variables (one path or glob per line)",
                html.Div(
                    [
                        dcc.Textarea(
                            id="ann-form-variables",
                            value="\n".join(ann.variables),
                            style={**_TEXTAREA, "height": "60px"},
                        ),
                        html.Div(
                            id="ann-form-varcount",
                            style={"fontSize": "11px", "color": _MUTED},
                        ),
                    ]
                ),
            ),
            html.Div(
                [
                    _field(
                        "Magnitude",
                        dcc.Input(
                            id="ann-form-magnitude", value=ann.magnitude, style=_INPUT
                        ),
                        "calc(33% - 8px)",
                    ),
                    _field(
                        "Cause",
                        dcc.Input(id="ann-form-cause", value=ann.cause, style=_INPUT),
                        "calc(33% - 8px)",
                    ),
                    _field(
                        "References (comma-separated)",
                        dcc.Input(
                            id="ann-form-refs", value=", ".join(ann.refs), style=_INPUT
                        ),
                        "calc(33% - 8px)",
                    ),
                ],
                style=row,
            ),
            html.Div(
                [
                    _field(
                        "Description (Markdown)",
                        dcc.Textarea(
                            id="ann-form-text",
                            value=ann.text,
                            style={**_TEXTAREA, "height": "140px"},
                        ),
                        "calc(50% - 6px)",
                    ),
                    _field(
                        "Preview",
                        html.Div(
                            dcc.Markdown(ann.text, id="ann-form-text-preview"),
                            style={
                                "border": "1px dashed #cfd8dc",
                                "borderRadius": "4px",
                                "padding": "0 10px",
                                "height": "140px",
                                "overflowY": "auto",
                                "fontSize": "13px",
                                "backgroundColor": "#fafafa",
                            },
                        ),
                        "calc(50% - 6px)",
                    ),
                ],
                style=row,
            ),
            _field(
                "Effect (Markdown, optional)",
                dcc.Textarea(
                    id="ann-form-effect",
                    value=ann.effect,
                    style={**_TEXTAREA, "height": "50px"},
                ),
            ),
            _field(
                "Figures",
                figure_table,
            ),
            html.Div(
                [
                    html.Button(
                        "\U0001f4be Save",
                        id="ann-form-save",
                        n_clicks=0,
                        title="Ctrl+S",
                        style=_BTN_PRIMARY,
                        **_CTRL_S,
                    ),
                    html.Button(
                        "Overwrite anyway",
                        id="ann-form-force",
                        n_clicks=0,
                        style={**_BTN_DANGER, "display": "none"},
                    ),
                    dcc.ConfirmDialogProvider(
                        html.Button(
                            "\U0001f5d1 Delete",
                            style={
                                **_BTN_DANGER,
                                "display": "inline-block" if original_id else "none",
                            },
                        ),
                        id="ann-form-delete",
                        message="Delete this annotation?",
                    ),
                    html.Button("Close", id="ann-form-cancel", n_clicks=0, style=_BTN),
                    html.Span(id="ann-form-msg", style={"fontSize": "12px"}),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center"},
            ),
        ],
        style={
            **_CARD,
            "borderColor": _ACCENT,
            "marginTop": "8px" if context == "drawer" else "0",
        },
    )


def build_section_form(
    store: AnnotationStore,
    section: Section | None,
    *,
    original_id: str | None,
) -> html.Div:
    """Edit form for one section of the outline."""
    doc = store.document
    sec = section or Section(id="")
    parent_id = None
    if original_id:
        for s, parent, _ in doc.iter_sections():
            if s.id == original_id:
                parent_id = parent.id if parent else None
    excluded = {original_id}
    if original_id:
        node = doc.section(original_id)

        def _collect(s: Section) -> None:
            for child in s.subsections:
                excluded.add(child.id)
                _collect(child)

        if node is not None:
            _collect(node)
    parent_options = [o for o in _section_options(doc) if o["value"] not in excluded]
    ann_options = [
        {"label": f"{a.title or a.id} ({a.id})", "value": a.id} for a in doc.annotations
    ]
    title = "New section" if original_id is None else f"Edit section “{original_id}”"
    row = {"display": "flex", "gap": "12px", "flexWrap": "wrap"}
    return html.Div(
        [
            dcc.Store(id="sec-form-original", data=original_id),
            dcc.Store(id="sec-form-revision", data=store.revision),
            html.H4(title, style={"margin": "0 0 8px 0"}),
            html.Div(
                [
                    _field(
                        "Title",
                        dcc.Input(id="sec-form-title", value=sec.title, style=_INPUT),
                        "60%",
                    ),
                    _field(
                        "Id",
                        dcc.Input(
                            id="sec-form-id",
                            value=sec.id,
                            placeholder="generated from the title",
                            style=_INPUT,
                        ),
                        "calc(40% - 12px)",
                    ),
                ],
                style=row,
            ),
            html.Div(
                [
                    _field(
                        "Parent section",
                        dcc.Dropdown(
                            id="sec-form-parent",
                            options=parent_options,
                            value=parent_id,
                            placeholder="Top level",
                        ),
                        "calc(50% - 6px)",
                    ),
                    _field(
                        "Generated content",
                        dcc.Dropdown(
                            id="sec-form-auto",
                            options=[
                                {"label": _AUTO_LABELS.get(a, a), "value": a}
                                for a in AUTO_SECTIONS
                            ],
                            value=sec.auto,
                            placeholder="None",
                        ),
                        "calc(50% - 6px)",
                    ),
                ],
                style=row,
            ),
            html.Div(
                [
                    _field(
                        "Body (Markdown)",
                        dcc.Textarea(
                            id="sec-form-body",
                            value=sec.body,
                            style={**_TEXTAREA, "height": "160px"},
                        ),
                        "calc(50% - 6px)",
                    ),
                    _field(
                        "Preview",
                        html.Div(
                            dcc.Markdown(sec.body, id="sec-form-body-preview"),
                            style={
                                "border": "1px dashed #cfd8dc",
                                "borderRadius": "4px",
                                "padding": "0 10px",
                                "height": "160px",
                                "overflowY": "auto",
                                "fontSize": "13px",
                                "backgroundColor": "#fafafa",
                            },
                        ),
                        "calc(50% - 6px)",
                    ),
                ],
                style=row,
            ),
            _field(
                "Annotations (in display order)",
                dcc.Dropdown(
                    id="sec-form-annotations",
                    options=ann_options,
                    value=list(sec.annotations),
                    multi=True,
                ),
            ),
            html.Div(
                [
                    html.Button(
                        "\U0001f4be Save",
                        id="sec-form-save",
                        n_clicks=0,
                        title="Ctrl+S",
                        style=_BTN_PRIMARY,
                        **_CTRL_S,
                    ),
                    dcc.ConfirmDialogProvider(
                        html.Button(
                            "\U0001f5d1 Delete",
                            style={
                                **_BTN_DANGER,
                                "display": "inline-block" if original_id else "none",
                            },
                        ),
                        id="sec-form-delete",
                        message="Delete this section and its subsections? "
                        "Annotations are kept.",
                    ),
                    html.Span(id="sec-form-msg", style={"fontSize": "12px"}),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center"},
            ),
        ],
        style={**_CARD, "borderColor": _ACCENT},
    )


# ── Editor page ─────────────────────────────────────────────────────


def build_outline(state: Any) -> list[Any]:
    """Outline tree (sections with their annotations) plus loose annotations."""
    store = get_store(state)
    if store is None:
        return []
    doc = store.document
    editable = not store.readonly
    cov_counts: dict[str, int] = {}
    for anns in coverage(state).values():
        for a in anns:
            cov_counts[a.id] = cov_counts.get(a.id, 0) + 1

    def _ann_item(ann_id: str, depth: int) -> html.Div:
        ann = doc.annotation(ann_id)
        if ann is None:
            return html.Div()
        label, color = _category(doc, ann.category)
        n = cov_counts.get(ann.id, 0)
        return html.Div(
            [
                html.Span("●", title=label, style={"color": color}),
                html.Button(
                    ann.title or ann.id,
                    id={"type": "outline-ann", "id": ann.id},
                    n_clicks=0,
                    style={
                        "border": "none",
                        "background": "none",
                        "cursor": "pointer",
                        "textAlign": "left",
                        "fontSize": "12px",
                        "padding": "1px 4px",
                        "flex": "1",
                    },
                ),
                html.Span(
                    f"{n} var" if n else "no match",
                    style={
                        "fontSize": "10px",
                        "color": _MUTED if n else "#c62828",
                        "whiteSpace": "nowrap",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "4px",
                "paddingLeft": f"{16 + depth * 16}px",
            },
        )

    items: list[Any] = []
    for section, _, depth in doc.iter_sections():
        controls: list[Any] = []
        if editable:
            controls = [
                html.Button(
                    sym,
                    id={"type": "outline-sec-move", "id": section.id, "delta": delta},
                    n_clicks=0,
                    title=tip,
                    style={**_BTN, "padding": "0 5px", "fontSize": "10px"},
                )
                for sym, delta, tip in (("▲", -1, "Move up"), ("▼", 1, "Move down"))
            ]
        auto = f" [{section.auto}]" if section.auto else ""
        items.append(
            html.Div(
                [
                    html.Button(
                        f"{section.title or section.id}{auto}",
                        id={"type": "outline-sec", "id": section.id},
                        n_clicks=0,
                        style={
                            "border": "none",
                            "background": "none",
                            "cursor": "pointer",
                            "textAlign": "left",
                            "fontWeight": "600",
                            "fontSize": "12px",
                            "flex": "1",
                            "padding": "2px 4px",
                        },
                    ),
                    *controls,
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "2px",
                    "paddingLeft": f"{depth * 16}px",
                    "marginTop": "4px",
                },
            )
        )
        items.extend(_ann_item(a, depth) for a in section.annotations)
    loose = doc.unplaced_annotations()
    if loose:
        items.append(
            html.Div(
                "Not placed",
                style={
                    "fontSize": "11px",
                    "color": _MUTED,
                    "fontStyle": "italic",
                    "marginTop": "8px",
                },
            )
        )
        items.extend(_ann_item(a.id, 0) for a in loose)
    if not items:
        items.append(
            html.Div(
                "The document is empty. Add a section or an annotation.",
                style={"fontSize": "12px", "color": _MUTED},
            )
        )
    return items


def _doc_form(store: AnnotationStore) -> html.Div:
    doc = store.document
    row = {"display": "flex", "gap": "12px", "flexWrap": "wrap"}
    editable = not store.readonly
    return html.Div(
        [
            dcc.Store(id="doc-form-revision", data=store.revision),
            html.Div(
                [
                    _field(
                        "Title",
                        dcc.Input(id="doc-form-title", value=doc.title, style=_INPUT),
                        "60%",
                    ),
                    _field(
                        "Date",
                        dcc.Input(id="doc-form-date", value=doc.date, style=_INPUT),
                        "calc(20% - 12px)",
                    ),
                    _field(
                        "Report language",
                        dcc.Dropdown(
                            id="doc-form-language",
                            options=[
                                {"label": lang, "value": lang} for lang in LANGUAGES
                            ],
                            value=doc.language,
                            clearable=False,
                        ),
                        "calc(20% - 12px)",
                    ),
                ],
                style=row,
            ),
            _field(
                "Subtitle",
                dcc.Input(id="doc-form-subtitle", value=doc.subtitle, style=_INPUT),
            ),
            html.Div(
                [
                    _field(
                        "Authors (comma-separated)",
                        dcc.Input(
                            id="doc-form-authors",
                            value=", ".join(doc.authors),
                            style=_INPUT,
                        ),
                        "calc(50% - 6px)",
                    ),
                    _field(
                        "Source label",
                        dcc.Input(
                            id="doc-form-source-label",
                            value=doc.source_label,
                            style=_INPUT,
                        ),
                        "calc(25% - 9px)",
                    ),
                    _field(
                        "Target label",
                        dcc.Input(
                            id="doc-form-target-label",
                            value=doc.target_label,
                            style=_INPUT,
                        ),
                        "calc(25% - 9px)",
                    ),
                ],
                style=row,
            ),
            _field(
                "Compared data (report header table)",
                build_row_editor(
                    "doc-form-datasets",
                    DATASET_COLUMNS,
                    [{"label": r.label, "value": r.value} for r in doc.datasets],
                    editable=editable,
                ),
            ),
            html.Div(
                [
                    html.Button(
                        "\U0001f4be Save",
                        id="doc-form-save",
                        n_clicks=0,
                        style=_BTN_PRIMARY,
                        **_CTRL_S,
                    ),
                    html.Span(id="doc-form-msg", style={"fontSize": "12px"}),
                ],
                style={
                    "display": "flex" if editable else "none",
                    "gap": "8px",
                    "alignItems": "center",
                },
            ),
        ],
        style=_CARD,
    )


def _categories_form(store: AnnotationStore) -> html.Div:
    doc = store.document
    editable = not store.readonly
    return html.Div(
        [
            dcc.Store(id="cat-form-revision", data=store.revision),
            build_row_editor(
                "cat-table",
                CATEGORY_COLUMNS,
                [
                    {
                        "id": c.id,
                        "label": c.label,
                        "color": c.color,
                        "description": c.description,
                    }
                    for c in doc.categories
                ],
                editable=editable,
                add_label="+ Add category",
            ),
            html.Div(
                [
                    html.Button(
                        "\U0001f4be Save",
                        id="cat-save",
                        n_clicks=0,
                        style=_BTN_PRIMARY,
                        **_CTRL_S,
                    ),
                    html.Span(id="cat-msg", style={"fontSize": "12px"}),
                ],
                style={
                    "display": "flex" if editable else "none",
                    "gap": "8px",
                    "alignItems": "center",
                    "marginTop": "8px",
                },
            ),
        ],
        style=_CARD,
    )


def build_annotations_view(state: Any) -> list[Any]:
    """Full-page annotation editor."""
    store = get_store(state)
    if store is None:
        return [html.Div("No annotation file was given (--annotations).")]
    editable = not store.readonly
    mode = "read-only" if store.readonly else "editable"
    header = html.Div(
        [
            html.H3("\U0001f4dd Annotations", style={"margin": "0", "flex": "1"}),
            html.Code(str(store.path), style={"fontSize": "11px"}),
            html.Span(f"({mode})", style={"fontSize": "11px", "color": _MUTED}),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "marginBottom": "10px",
        },
    )
    outline_actions: list[Any] = []
    if editable:
        outline_actions = [
            html.Button("+ Section", id="outline-new-sec", n_clicks=0, style=_BTN),
            html.Button("+ Annotation", id="outline-new-ann", n_clicks=0, style=_BTN),
        ]
    outline_tab = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        outline_actions,
                        style={"display": "flex", "gap": "6px", "marginBottom": "8px"},
                    ),
                    html.Div(build_outline(state), id="ann-outline-pane"),
                    html.Div(id="outline-msg", style={"fontSize": "12px"}),
                ],
                style={
                    **_CARD,
                    "width": "320px",
                    "flexShrink": 0,
                    "maxHeight": "75vh",
                    "overflowY": "auto",
                },
            ),
            html.Div(
                html.Div(
                    "Select a section or an annotation in the outline.",
                    style={"color": _MUTED, "fontSize": "13px", "padding": "12px"},
                ),
                id="ann-editor-pane",
                style={"flex": "1", "minWidth": "0"},
            ),
        ],
        style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
    )
    tab_style = {"padding": "6px 14px", "fontSize": "13px"}
    return [
        header,
        dcc.Tabs(
            [
                dcc.Tab(
                    outline_tab,
                    label="Outline & annotations",
                    style=tab_style,
                    selected_style={**tab_style, "fontWeight": "600"},
                ),
                dcc.Tab(
                    _doc_form(store),
                    label="Document",
                    style=tab_style,
                    selected_style={**tab_style, "fontWeight": "600"},
                ),
                dcc.Tab(
                    _categories_form(store),
                    label="Categories",
                    style=tab_style,
                    selected_style={**tab_style, "fontWeight": "600"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),
    ]
