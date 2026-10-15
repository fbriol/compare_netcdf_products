"""Callbacks of the annotation display and editor.

Every mutation goes through :meth:`AnnotationStore.apply`, which
validates the document and writes the JSON file atomically. After a
successful change the global ``ann-revision`` store is bumped; the
outline, the detail-view panel and the forms listen to it to refresh.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from dash import ALL, Input, Output, State, callback_context, html, no_update

from ..annotated_report import ReportInputs, build_report
from ..annotations import (
    Annotation,
    AnnotationError,
    Category,
    ConflictError,
    DatasetRow,
    FigureSpec,
    ReadOnlyError,
    Section,
    delete_annotation,
    delete_section,
    move_section,
    pattern_matches,
    unique_id,
    upsert_annotation,
    upsert_section,
)
from ._registration import _apply_expression_no_nav, _make_banner
from .annotation_views import (
    CATEGORY_COLUMNS,
    DATASET_COLUMNS,
    FIGURE_COLUMNS,
    all_varpaths,
    build_annotation_cards,
    build_annotation_form,
    build_outline,
    build_section_form,
    get_store,
    rows_to_figures,
)
from .row_editor import cell_states, collect_rows, register_row_editor
from .views import render_current_view

if TYPE_CHECKING:
    import dash

    from ..state import SessionState

LOGGER = logging.getLogger(__name__)

_OK = {"color": "#2e7d32"}
_ERR = {"color": "#c62828"}
_FORCE_SHOWN = {
    "border": "1px solid #ef9a9a",
    "borderRadius": "4px",
    "padding": "4px 10px",
    "backgroundColor": "white",
    "cursor": "pointer",
    "fontSize": "12px",
    "color": "#c62828",
}
_HIDDEN = {"display": "none"}


def _clicked() -> dict[str, Any] | str | None:
    """Return the id of the component actually clicked, if any.

    Pattern-matching buttons fire with ``n_clicks == 0`` when they are
    (re)rendered; those triggers are ignored.
    """
    best_val, best_id = 0, None
    for trig in callback_context.triggered:
        val = trig.get("value") or 0
        if val > best_val:
            best_val = val
            raw = trig["prop_id"].rsplit(".", 1)[0]
            best_id = json.loads(raw) if raw.startswith("{") else raw
    return best_id


def _error_message(exc: Exception) -> html.Span:
    if isinstance(exc, AnnotationError):
        return html.Span(
            ["⚠ Not saved:", html.Ul([html.Li(e) for e in exc.errors])],
            style=_ERR,
        )
    return html.Span(f"⚠ {exc}", style=_ERR)


def _lines(text: str | None) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _csv(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def _zoom_from_store(zoom_range: dict | None) -> list[float] | None:
    if not zoom_range or zoom_range.get("auto"):
        return None
    lo, hi = zoom_range.get("min"), zoom_range.get("max")
    if lo is None or hi is None:
        return None
    return [round(float(lo), 3), round(float(hi), 3)]


def _follow_revision(tab_revision, form_revision):
    """Advance a form's base revision to the one this tab just saved."""
    if tab_revision is None or (form_revision or 0) >= tab_revision:
        return no_update
    return tab_revision


def register_annotation_callbacks(app: dash.Dash, state: SessionState) -> None:
    """Register the callbacks of the annotation features."""

    # ── Navigation to the editor page ─────────────────────────────

    @app.callback(
        Output("nav-current-view", "data", allow_duplicate=True),
        Input("annotations-nav-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_editor(n_clicks):
        if not n_clicks:
            return no_update
        return "annotations"

    # ── Detail view: open the drawer ──────────────────────────────

    @app.callback(
        Output("ann-drawer", "children"),
        Input("ann-new-for-var", "n_clicks"),
        Input({"type": "ann-edit", "id": ALL}, "n_clicks"),
        State("nav-varpath", "data"),
        State("zoom-range", "data"),
        State("plot-visibility", "value"),
        prevent_initial_call=True,
    )
    def open_drawer(_n_new, _n_edit, varpath, zoom_range, visible):
        store = get_store(state)
        clicked = _clicked()
        if store is None or store.readonly or clicked is None:
            return no_update
        if isinstance(clicked, dict):
            ann = store.document.annotation(clicked["id"])
            if ann is None:
                return no_update
            return build_annotation_form(
                store, ann, original_id=ann.id, context="drawer"
            )
        figure = FigureSpec(
            variable=varpath,
            plots=list(visible or []),
            expression=state.active_expression or "",
            zoom=_zoom_from_store(zoom_range),
        )
        draft = Annotation(id="", variables=[varpath], figures=[figure])
        return build_annotation_form(store, draft, original_id=None, context="drawer")

    @app.callback(
        Output("ann-drawer", "children", allow_duplicate=True),
        Input("ann-form-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_drawer(n_clicks):
        return [] if n_clicks else no_update

    @app.callback(
        Output("ann-editor-pane", "children", allow_duplicate=True),
        Input("ann-form-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_editor_form(n_clicks):
        return [] if n_clicks else no_update

    # ── Editor page: pick an outline item ─────────────────────────

    @app.callback(
        Output("ann-editor-pane", "children"),
        Input({"type": "outline-ann", "id": ALL}, "n_clicks"),
        Input({"type": "outline-sec", "id": ALL}, "n_clicks"),
        Input("outline-new-ann", "n_clicks"),
        Input("outline-new-sec", "n_clicks"),
        prevent_initial_call=True,
    )
    def pick_outline_item(_a, _s, _new_a, _new_s):
        store = get_store(state)
        clicked = _clicked()
        if store is None or clicked is None:
            return no_update
        doc = store.document
        if clicked == "outline-new-ann":
            return build_annotation_form(store, None, original_id=None)
        if clicked == "outline-new-sec":
            return build_section_form(store, None, original_id=None)
        if not isinstance(clicked, dict):
            return no_update
        if clicked["type"] == "outline-ann":
            ann = doc.annotation(clicked["id"])
            if ann is None:
                return no_update
            if store.readonly:
                from .annotation_views import annotation_card

                return annotation_card(doc, ann, editable=False)
            return build_annotation_form(store, ann, original_id=ann.id)
        section = doc.section(clicked["id"])
        if section is None:
            return no_update
        if store.readonly:
            return html.Div(
                [html.H4(section.title), html.Pre(section.body)],
                style={"padding": "8px"},
            )
        return build_section_form(store, section, original_id=section.id)

    @app.callback(
        Output("ann-revision", "data", allow_duplicate=True),
        Output("outline-msg", "children"),
        Input({"type": "outline-sec-move", "id": ALL, "delta": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def move_outline_section(_n):
        store = get_store(state)
        clicked = _clicked()
        if store is None or not isinstance(clicked, dict):
            return no_update, no_update
        try:
            rev = store.apply(
                lambda d: move_section(d, clicked["id"], int(clicked["delta"]))
            )
        except (AnnotationError, ConflictError, ReadOnlyError) as exc:
            return no_update, _error_message(exc)
        return rev, ""

    # ── Refresh views after a change ──────────────────────────────

    @app.callback(
        Output("ann-outline-pane", "children"),
        Input("ann-revision", "data"),
        prevent_initial_call=True,
    )
    def refresh_outline(_rev):
        return build_outline(state)

    @app.callback(
        Output("annotation-panel-cards", "children"),
        Input("ann-revision", "data"),
        State("nav-varpath", "data"),
        prevent_initial_call=True,
    )
    def refresh_panel(_rev, varpath):
        if not varpath:
            return no_update
        return build_annotation_cards(state, varpath)

    # Forms of this tab follow the revisions this tab produced, so only
    # saves made from another tab (or on disk) are reported as conflicts.
    for form_store in (
        "ann-form-revision",
        "sec-form-revision",
        "doc-form-revision",
        "cat-form-revision",
    ):
        app.callback(
            Output(form_store, "data", allow_duplicate=True),
            Input("ann-revision", "data"),
            State(form_store, "data"),
            prevent_initial_call=True,
        )(_follow_revision)

    # ── Annotation form helpers ───────────────────────────────────

    @app.callback(
        Output("ann-form-text-preview", "children"),
        Input("ann-form-text", "value"),
        prevent_initial_call=True,
    )
    def preview_text(text):
        return text or ""

    @app.callback(
        Output("sec-form-body-preview", "children"),
        Input("sec-form-body", "value"),
        prevent_initial_call=True,
    )
    def preview_body(text):
        return text or ""

    @app.callback(
        Output("ann-form-varcount", "children"),
        Input("ann-form-variables", "value"),
    )
    def count_matches(text):
        patterns = _lines(text)
        if not patterns:
            return "No variable."
        varpaths = all_varpaths(state)
        matched = [v for v in varpaths if any(pattern_matches(p, v) for p in patterns)]
        dead = [p for p in patterns if not any(pattern_matches(p, v) for v in varpaths)]
        msg = f"{len(matched)} variable(s) matched"
        if dead:
            return html.Span(f"{msg} — no match for: {', '.join(dead)}", style=_ERR)
        return msg

    def _blank_figure(variables):
        exact = [v for v in _lines(variables) if not set(v) & set("*?[")]
        return {"variable": exact[0] if exact else ""}

    register_row_editor(
        app,
        "ann-form-figures",
        FIGURE_COLUMNS,
        _blank_figure,
        [State("ann-form-variables", "value")],
    )
    register_row_editor(
        app, "doc-form-datasets", DATASET_COLUMNS, lambda: {"label": "", "value": ""}
    )
    register_row_editor(
        app,
        "cat-table",
        CATEGORY_COLUMNS,
        lambda: {"id": "", "label": "", "color": "#607d8b", "description": ""},
    )

    app.clientside_callback(
        """
        function() {
            var ctx = window.dash_clientside.callback_context;
            var t = (ctx.triggered[0] || {}).prop_id || '';
            if (t.indexOf('ann-form-revision') === 0) return '';
            return '\\u25cf Unsaved changes';
        }
        """,
        Output("ann-form-dirty", "children"),
        Input("ann-form-title", "value"),
        Input("ann-form-id", "value"),
        Input("ann-form-category", "value"),
        Input("ann-form-status", "value"),
        Input("ann-form-section", "value"),
        Input("ann-form-variables", "value"),
        Input("ann-form-magnitude", "value"),
        Input("ann-form-cause", "value"),
        Input("ann-form-refs", "value"),
        Input("ann-form-text", "value"),
        Input("ann-form-effect", "value"),
        Input({"type": "ann-form-figures-cell", "row": ALL, "col": ALL}, "value"),
        Input("ann-form-revision", "data"),
        prevent_initial_call=True,
    )

    # ── Annotation form: save / overwrite / delete ────────────────

    @app.callback(
        Output("ann-form-msg", "children"),
        Output("ann-form-force", "style"),
        Output("ann-form-original", "data"),
        Output("ann-form-revision", "data"),
        Output("ann-form-id", "value"),
        Output("ann-revision", "data", allow_duplicate=True),
        Input("ann-form-save", "n_clicks"),
        Input("ann-form-force", "n_clicks"),
        Input("ann-form-delete", "submit_n_clicks"),
        State("ann-form-original", "data"),
        State("ann-form-revision", "data"),
        State("ann-form-id", "value"),
        State("ann-form-title", "value"),
        State("ann-form-category", "value"),
        State("ann-form-status", "value"),
        State("ann-form-section", "value"),
        State("ann-form-variables", "value"),
        State("ann-form-magnitude", "value"),
        State("ann-form-cause", "value"),
        State("ann-form-refs", "value"),
        State("ann-form-text", "value"),
        State("ann-form-effect", "value"),
        *cell_states("ann-form-figures"),
        prevent_initial_call=True,
    )
    def save_annotation(
        _save,
        _force,
        _delete,
        original_id,
        base_revision,
        ann_id,
        title,
        category,
        status,
        section_id,
        variables,
        magnitude,
        cause,
        refs,
        text,
        effect,
        figure_values,
        figure_ids,
    ):
        nothing = (no_update,) * 6
        store = get_store(state)
        clicked = _clicked()
        if store is None or clicked is None:
            return nothing
        force = clicked == "ann-form-force"

        if clicked == "ann-form-delete":
            if not original_id:
                return nothing
            try:
                rev = store.apply(
                    lambda d: delete_annotation(d, original_id),
                    base_revision=base_revision,
                )
            except (AnnotationError, ConflictError, ReadOnlyError) as exc:
                return _error_message(exc), no_update, *(no_update,) * 4
            msg = html.Span(f"✓ Deleted “{original_id}”", style=_OK)
            return msg, _HIDDEN, None, rev, no_update, rev

        figures, errors = rows_to_figures(collect_rows(figure_values, figure_ids))
        if errors:
            return _error_message(AnnotationError(errors)), *(no_update,) * 5
        existing = [a.id for a in store.document.annotations if a.id != original_id]
        new_id = (ann_id or "").strip() or unique_id(
            title or (_lines(variables) or ["annotation"])[0], existing
        )
        annotation = Annotation(
            id=new_id,
            title=(title or "").strip(),
            category=category or "",
            status=status or "open",
            variables=_lines(variables),
            text=text or "",
            effect=effect or "",
            magnitude=(magnitude or "").strip(),
            cause=(cause or "").strip(),
            refs=_csv(refs),
            figures=figures,
        )
        if original_id is None and new_id in existing:
            return (
                _error_message(AnnotationError([f"id {new_id!r} already exists"])),
                *(no_update,) * 5,
            )

        def _edit(doc):
            upsert_annotation(doc, annotation, previous_id=original_id)
            for section, _, _ in doc.iter_sections():
                if section.id != section_id and new_id in section.annotations:
                    section.annotations.remove(new_id)
            if section_id:
                target = doc.section(section_id)
                if target is not None and new_id not in target.annotations:
                    target.annotations.append(new_id)

        try:
            rev = store.apply(
                _edit,
                base_revision=None if force else base_revision,
                force=force,
            )
        except ConflictError as exc:
            hint = " Reload the annotations to merge, or overwrite them."
            return (
                html.Span(f"⚠ {exc}{hint}", style=_ERR),
                _FORCE_SHOWN,
                *(no_update,) * 4,
            )
        except (AnnotationError, ReadOnlyError) as exc:
            return _error_message(exc), *(no_update,) * 5
        msg = html.Span(f"✓ Saved to {store.path.name}", style=_OK)
        return msg, _HIDDEN, new_id, rev, new_id, rev

    # ── Detail view: append the current view to an annotation ─────

    @app.callback(
        Output("ann-panel-msg", "children"),
        Output("ann-revision", "data", allow_duplicate=True),
        Input({"type": "ann-addfig", "id": ALL}, "n_clicks"),
        State("nav-varpath", "data"),
        State("zoom-range", "data"),
        State("plot-visibility", "value"),
        prevent_initial_call=True,
    )
    def add_current_view(_n, varpath, zoom_range, visible):
        store = get_store(state)
        clicked = _clicked()
        if store is None or not isinstance(clicked, dict) or not varpath:
            return no_update, no_update
        figure = FigureSpec(
            variable=varpath,
            plots=list(visible or []),
            expression=state.active_expression or "",
            zoom=_zoom_from_store(zoom_range),
        )

        def _edit(doc):
            ann = doc.annotation(clicked["id"])
            if ann is not None:
                ann.figures.append(figure)

        try:
            rev = store.apply(_edit)
        except (AnnotationError, ConflictError, ReadOnlyError) as exc:
            return _error_message(exc), no_update
        return html.Span("✓ Figure added", style=_OK), rev

    # ── Detail view: reproduce a figure's view ────────────────────

    @app.callback(
        Output("breadcrumb-bar", "children", allow_duplicate=True),
        Output("main-content", "children", allow_duplicate=True),
        Output("zoom-range", "data", allow_duplicate=True),
        Output("expr-textarea", "value", allow_duplicate=True),
        Output("expr-status", "children", allow_duplicate=True),
        Output("expr-status", "style", allow_duplicate=True),
        Output("filter-banner", "children", allow_duplicate=True),
        Output("filter-banner", "style", allow_duplicate=True),
        Input({"type": "ann-showfig", "ann": ALL, "idx": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def show_figure_view(_n):
        store = get_store(state)
        clicked = _clicked()
        if store is None or not isinstance(clicked, dict):
            return (no_update,) * 8
        ann = store.document.annotation(clicked["ann"])
        if ann is None or clicked["idx"] >= len(ann.figures):
            return (no_update,) * 8
        spec = ann.figures[clicked["idx"]]
        status_msg: Any = no_update
        status_style: Any = no_update
        expr_value: Any = no_update
        if spec.expression:
            result = _apply_expression_no_nav(state, spec.expression)
            status_msg, status_style = result
            expr_value = spec.expression
        zoom = (
            {"min": spec.zoom[0], "max": spec.zoom[1]}
            if spec.zoom is not None
            else {"auto": True}
        )
        state.current_view = "detail"
        state.current_varpath = spec.variable
        bc, content = render_current_view(
            state, zoom_range=zoom, visible_plots=spec.plots or None
        )
        banner = _make_banner(state)
        return bc, content, zoom, expr_value, status_msg, status_style, *banner

    # ── Section form ──────────────────────────────────────────────

    @app.callback(
        Output("sec-form-msg", "children"),
        Output("sec-form-original", "data"),
        Output("sec-form-revision", "data"),
        Output("sec-form-id", "value"),
        Output("ann-revision", "data", allow_duplicate=True),
        Input("sec-form-save", "n_clicks"),
        Input("sec-form-delete", "submit_n_clicks"),
        State("sec-form-original", "data"),
        State("sec-form-revision", "data"),
        State("sec-form-id", "value"),
        State("sec-form-title", "value"),
        State("sec-form-parent", "value"),
        State("sec-form-auto", "value"),
        State("sec-form-body", "value"),
        State("sec-form-annotations", "value"),
        prevent_initial_call=True,
    )
    def save_section(
        _save,
        _delete,
        original_id,
        base_revision,
        sec_id,
        title,
        parent_id,
        auto,
        body,
        annotations,
    ):
        store = get_store(state)
        clicked = _clicked()
        if store is None or clicked is None:
            return (no_update,) * 5
        if clicked == "sec-form-delete":
            if not original_id:
                return (no_update,) * 5
            try:
                rev = store.apply(
                    lambda d: delete_section(d, original_id),
                    base_revision=base_revision,
                )
            except (AnnotationError, ConflictError, ReadOnlyError) as exc:
                return _error_message(exc), *(no_update,) * 4
            return html.Span("✓ Deleted", style=_OK), None, rev, no_update, rev

        existing = [
            s.id for s, _, _ in store.document.iter_sections() if s.id != original_id
        ]
        new_id = (sec_id or "").strip() or unique_id(title or "section", existing)
        section = Section(
            id=new_id,
            title=(title or "").strip(),
            body=body or "",
            annotations=list(annotations or []),
            auto=auto or None,
        )

        def _edit(doc):
            if original_id is not None:
                current_parent = None
                for s, parent, _ in doc.iter_sections():
                    if s.id == original_id:
                        current_parent = parent.id if parent else None
                if current_parent != parent_id:
                    node = doc.section(original_id)
                    if node is not None:
                        section.subsections = node.subsections
                    delete_section(doc, original_id)
                    upsert_section(doc, section, parent_id=parent_id)
                    return
            upsert_section(doc, section, previous_id=original_id, parent_id=parent_id)

        try:
            rev = store.apply(_edit, base_revision=base_revision)
        except (AnnotationError, ConflictError, ReadOnlyError) as exc:
            return _error_message(exc), *(no_update,) * 4
        msg = html.Span(f"✓ Saved to {store.path.name}", style=_OK)
        return msg, new_id, rev, new_id, rev

    # ── Document metadata ─────────────────────────────────────────

    @app.callback(
        Output("doc-form-msg", "children"),
        Output("doc-form-revision", "data"),
        Output("ann-revision", "data", allow_duplicate=True),
        Input("doc-form-save", "n_clicks"),
        State("doc-form-revision", "data"),
        State("doc-form-title", "value"),
        State("doc-form-subtitle", "value"),
        State("doc-form-date", "value"),
        State("doc-form-language", "value"),
        State("doc-form-authors", "value"),
        State("doc-form-source-label", "value"),
        State("doc-form-target-label", "value"),
        *cell_states("doc-form-datasets"),
        prevent_initial_call=True,
    )
    def save_document(
        n_clicks,
        base_revision,
        title,
        subtitle,
        date,
        language,
        authors,
        source_label,
        target_label,
        dataset_values,
        dataset_ids,
    ):
        store = get_store(state)
        if store is None or not n_clicks:
            return no_update, no_update, no_update

        def _edit(doc):
            doc.title = (title or "").strip()
            doc.subtitle = (subtitle or "").strip()
            doc.date = (date or "").strip()
            doc.language = language or "en"
            doc.authors = _csv(authors)
            doc.source_label = (source_label or "").strip() or "Source"
            doc.target_label = (target_label or "").strip() or "Target"
            doc.datasets = [
                DatasetRow(str(r.get("label") or ""), str(r.get("value") or ""))
                for r in collect_rows(dataset_values, dataset_ids)
                if r.get("label") or r.get("value")
            ]

        try:
            rev = store.apply(_edit, base_revision=base_revision)
        except (AnnotationError, ConflictError, ReadOnlyError) as exc:
            return _error_message(exc), no_update, no_update
        return html.Span(f"✓ Saved to {store.path.name}", style=_OK), rev, rev

    # ── Categories ────────────────────────────────────────────────

    @app.callback(
        Output("cat-msg", "children"),
        Output("cat-form-revision", "data"),
        Output("ann-revision", "data", allow_duplicate=True),
        Input("cat-save", "n_clicks"),
        State("cat-form-revision", "data"),
        *cell_states("cat-table"),
        prevent_initial_call=True,
    )
    def save_categories(n_clicks, base_revision, cat_values, cat_ids):
        store = get_store(state)
        if store is None or not n_clicks:
            return no_update, no_update, no_update
        categories = [
            Category(
                id=str(r.get("id") or "").strip(),
                label=str(r.get("label") or "").strip(),
                color=str(r.get("color") or "").strip() or "#607d8b",
                description=str(r.get("description") or ""),
            )
            for r in collect_rows(cat_values, cat_ids)
            if r.get("id") or r.get("label")
        ]
        used = {a.category for a in store.document.annotations if a.category}
        missing = sorted(used - {c.id for c in categories})
        if missing:
            return (
                _error_message(
                    AnnotationError(
                        [
                            f"category {m!r} is still used by annotations"
                            for m in missing
                        ]
                    )
                ),
                no_update,
                no_update,
            )

        def _edit(doc):
            doc.categories = categories

        try:
            rev = store.apply(_edit, base_revision=base_revision)
        except (AnnotationError, ConflictError, ReadOnlyError) as exc:
            return _error_message(exc), no_update, no_update
        return html.Span(f"✓ Saved to {store.path.name}", style=_OK), rev, rev

    # ── Sidebar: reload & export ──────────────────────────────────

    @app.callback(
        Output("ann-sidebar-msg", "children"),
        Output("ann-revision", "data", allow_duplicate=True),
        Output("breadcrumb-bar", "children", allow_duplicate=True),
        Output("main-content", "children", allow_duplicate=True),
        Input("ann-reload-btn", "n_clicks"),
        State("zoom-range", "data"),
        prevent_initial_call=True,
    )
    def reload_annotations(n_clicks, zoom_range):
        store = get_store(state)
        if store is None or not n_clicks:
            return (no_update,) * 4
        try:
            store.reload()
        except AnnotationError as exc:
            return _error_message(exc), no_update, no_update, no_update
        bc, content = render_current_view(state, zoom_range=zoom_range)
        msg = html.Span(
            f"✓ Reloaded ({len(store.document.annotations)} annotations)",
            style=_OK,
        )
        return msg, store.revision, bc, content

    @app.callback(
        Output("export-annotated-download", "data"),
        Output("ann-sidebar-msg", "children", allow_duplicate=True),
        Input("export-annotated-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_annotated(n_clicks):
        store = get_store(state)
        if store is None or not n_clicks:
            return no_update, no_update
        inputs = ReportInputs(
            src_ds=state.src_ds,
            tgt_ds=state.tgt_ds,
            src_name=state.src_path.name,
            tgt_name=state.tgt_path.name,
            results=state.group_results,
            flags=state.flags,
            mask=state.active_mask,
            expression=state.active_expression,
            tolerance_steps=state.tolerance_steps,
        )
        try:
            page = build_report(store.snapshot(), inputs)
        except Exception as exc:
            LOGGER.exception("Annotated report failed")
            return no_update, html.Span(f"⚠ {exc}", style=_ERR)
        return (
            {
                "content": page,
                "filename": f"{store.path.stem}_report.html",
                "type": "text/html",
            },
            html.Span("✓ Report generated", style=_OK),
        )
