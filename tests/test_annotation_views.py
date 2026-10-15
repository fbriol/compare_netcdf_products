"""Tests for the Dash annotation components and the views using them."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.annotations import Annotation, AnnotationStore, FigureSpec, upsert_annotation
from app.callbacks.annotation_views import (
    build_annotation_form,
    build_annotation_overview,
    build_annotation_panel,
    build_annotations_view,
    figures_to_rows,
    rows_to_figures,
)
from app.callbacks.views import render_current_view
from app.comparison import scan_all_groups
from app.layout import create_layout
from app.state import SessionState
from tests.test_layout import _collect_ids


@pytest.fixture()
def annotated_state(open_nc_pair, tmp_path: Path):
    src, tgt = open_nc_pair
    store = AnnotationStore(tmp_path / "ann.json")
    store.apply(
        lambda d: upsert_annotation(
            d,
            Annotation(
                id="speed",
                title="Speed differs",
                category="action",
                variables=["/data_01/speed"],
                figures=[FigureSpec("/data_01/speed", ["Histogram"], zoom=[1, 4])],
            ),
        )
    )
    state = SessionState(
        src_path=tmp_path / "ref.nc",
        tgt_path=tmp_path / "tgt.nc",
        src_ds=src,
        tgt_ds=tgt,
        annotations=store,
    )
    state.group_results = scan_all_groups(src, tgt, None)
    return state


def test_layout_has_annotation_controls(annotated_state) -> None:
    ids = _collect_ids(create_layout(annotated_state))
    for required in {
        "ann-revision",
        "annotations-nav-btn",
        "export-annotated-btn",
        "ann-reload-btn",
    }:
        assert required in ids


def test_overview_counts_coverage(annotated_state) -> None:
    overview = build_annotation_overview(annotated_state)
    assert "1/2 differing variables annotated" in str(overview)


def test_detail_view_contains_panel(annotated_state) -> None:
    annotated_state.current_view = "detail"
    annotated_state.current_varpath = "/data_01/speed"
    _, content = render_current_view(annotated_state, visible_plots=["Histogram"])
    text = str(content)
    assert "annotation-panel" in text and "Speed differs" in text
    assert "ncj-fullscreen-btn" in text
    assert "ann-new-for-var" in text
    panel = build_annotation_panel(annotated_state, "/data_01/altitude")
    assert "No annotation covers" in str(panel)


def test_editor_view_and_forms(annotated_state) -> None:
    annotated_state.current_view = "annotations"
    _, content = render_current_view(annotated_state)
    ids = set()
    for component in content:
        ids |= _collect_ids(component)
    assert {"ann-outline-pane", "ann-editor-pane", "doc-form-save", "cat-table"} <= ids
    store = annotated_state.annotations
    form = build_annotation_form(
        store, store.document.annotation("speed"), original_id="speed"
    )
    assert {"ann-form-save", "ann-form-figures", "ann-form-figures-add"} <= (
        _collect_ids(form)
    )


def test_readonly_hides_editing(annotated_state, tmp_path: Path) -> None:
    ro = AnnotationStore(tmp_path / "ann.json", readonly=True)
    object.__setattr__(annotated_state.ctx, "annotations", ro)
    panel = build_annotation_panel(annotated_state, "/data_01/speed")
    assert "ann-new-for-var" not in str(panel)
    assert "ann-edit" not in str(panel)
    view = build_annotations_view(annotated_state)
    assert "outline-new-ann" not in str(view)


def test_figure_rows_roundtrip() -> None:
    figures = [FigureSpec("/a", ["Histogram", "Statistics"], "${/a} > 0", [1, 2.5])]
    parsed, errors = rows_to_figures(figures_to_rows(figures))
    assert errors == [] and parsed == [
        FigureSpec("/a", ["Histogram", "Statistics"], "${/a} > 0", [1.0, 2.5])
    ]
    _, errors = rows_to_figures([{"variable": "/a", "zoom": "nope"}])
    assert errors
    assert rows_to_figures([{"variable": ""}]) == ([], [])


def test_collect_rows_orders_by_row() -> None:
    from app.callbacks.row_editor import collect_rows

    ids = [
        {"type": "t-cell", "row": 1, "col": "a"},
        {"type": "t-cell", "row": 0, "col": "a"},
        {"type": "t-cell", "row": 0, "col": "b"},
    ]
    assert collect_rows(["x", "y", "z"], ids) == [{"a": "y", "b": "z"}, {"a": "x"}]
    assert collect_rows(None, None) == []


def _find(component, cid):
    if getattr(component, "id", None) == cid:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    for child in children if isinstance(children, (list, tuple)) else [children]:
        found = _find(child, cid)
        if found is not None:
            return found
    return None


def test_detail_view_shows_difference_by_default(annotated_state) -> None:
    annotated_state.current_view = "detail"
    annotated_state.current_varpath = "/data_01/speed"
    _, content = render_current_view(annotated_state)
    found = (_find(x, "plot-visibility") for x in content)
    pills = next(c for c in found if c is not None)
    assert pills.value == ["Signal Difference"]
    containers = {
        c.id["label"]: c.style
        for c in content
        if isinstance(getattr(c, "id", None), dict)
        and c.id.get("type") == "plot-container"
    }
    assert containers["Signal Comparison"] == {"display": "none"}
    assert containers["Signal Difference"] != {"display": "none"}
