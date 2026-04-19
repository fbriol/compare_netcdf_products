"""Smoke tests for app/layout.py — guard against silent ID drift."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.layout import create_layout


def _collect_ids(component) -> set[str]:
    ids: set[str] = set()
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        ids.add(cid)
    children = getattr(component, "children", None)
    if children is None:
        return ids
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "id") or hasattr(child, "children"):
            ids |= _collect_ids(child)
    return ids


def test_layout_contains_required_store_ids() -> None:
    state = MagicMock()
    state.src_path.name = "src.nc"
    state.tgt_path.name = "tgt.nc"
    state.flags = None
    state.active_expression = None
    state.filter_info = None
    state.group_results = []

    layout = create_layout(state)
    ids = _collect_ids(layout)
    for required in {
        "filter-conditions",
        "zoom-range",
        "nav-varpath",
        "export-btn",
        "export-csv-btn",
        "export-json-btn",
    }:
        assert required in ids, f"missing layout id: {required}"
