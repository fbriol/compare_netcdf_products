"""Tests for the on-disk annotation store used by the editor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.annotations import (
    Annotation,
    AnnotationError,
    AnnotationStore,
    ConflictError,
    ReadOnlyError,
    upsert_annotation,
)


def test_missing_file_is_created(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    store = AnnotationStore(path)
    assert path.exists()
    assert json.loads(path.read_text())["categories"]
    assert not store.disk_changed()


def test_missing_file_readonly_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AnnotationStore(tmp_path / "ann.json", readonly=True)


def test_apply_roundtrip_and_backup(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    store = AnnotationStore(path)
    rev = store.revision
    before = path.read_text()
    new_rev = store.apply(
        lambda d: upsert_annotation(d, Annotation(id="a1", title="Été")),
        base_revision=rev,
    )
    assert new_rev == rev + 1
    assert (tmp_path / "ann.json.bak").read_text() == before
    assert not (tmp_path / "ann.json.tmp").exists()
    assert "Été" in path.read_text(encoding="utf-8")
    reloaded = AnnotationStore(path)
    assert reloaded.document.annotation("a1").title == "Été"


def test_stale_revision_conflict(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "ann.json")
    rev = store.revision
    store.apply(lambda d: upsert_annotation(d, Annotation(id="a1")), base_revision=rev)
    with pytest.raises(ConflictError, match="another tab"):
        store.apply(
            lambda d: upsert_annotation(d, Annotation(id="a2")), base_revision=rev
        )


def test_disk_change_conflict_and_force(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    store = AnnotationStore(path)
    raw = json.loads(path.read_text())
    raw["title"] = "edited by hand"
    path.write_text(json.dumps(raw))
    assert store.disk_changed()
    with pytest.raises(ConflictError, match="modified on disk"):
        store.apply(lambda d: setattr(d, "title", "ui"))
    store.apply(lambda d: setattr(d, "title", "ui"), force=True)
    assert json.loads(path.read_text())["title"] == "ui"
    path.write_text(json.dumps(raw))
    store.reload()
    assert store.document.title == "edited by hand"


def test_invalid_edit_is_not_written(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    store = AnnotationStore(path)
    before = path.read_text()
    with pytest.raises(AnnotationError, match="unknown category"):
        store.apply(lambda d: upsert_annotation(d, Annotation(id="a", category="nope")))
    assert path.read_text() == before
    assert store.document.annotation("a") is None


def test_invalid_reload_keeps_document(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    store = AnnotationStore(path)
    path.write_text("{oops")
    with pytest.raises(AnnotationError):
        store.reload()
    assert store.document.categories


def test_readonly(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    AnnotationStore(path)
    store = AnnotationStore(path, readonly=True)
    with pytest.raises(ReadOnlyError):
        store.apply(lambda d: None)
