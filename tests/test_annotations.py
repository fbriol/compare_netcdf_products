"""Tests for the annotation document model and loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.annotations import (
    Annotation,
    AnnotationError,
    FigureSpec,
    Section,
    check_against_results,
    delete_annotation,
    document_from_dict,
    dump_annotations,
    json_schema,
    load_annotations,
    main,
    match_annotations,
    move_section,
    new_document,
    template_from_results,
    unique_id,
    upsert_annotation,
    upsert_section,
)
from app.comparison import scan_all_groups
from app.constants import PLOT_LABELS


def _raw() -> dict:
    return {
        "schema_version": 1,
        "title": "Validation",
        "source_label": "PDAP",
        "target_label": "c3po",
        "datasets": [{"label": "Pass", "value": "cycle 188"}],
        "categories": [{"id": "tds", "label": "TDS", "color": "#c62828"}],
        "sections": [
            {
                "id": "s1",
                "title": "Points",
                "subsections": [{"id": "s1-1", "title": "L1B", "annotations": ["a1"]}],
            },
            {"id": "acc", "title": "Agreement", "auto": "agreement_table"},
        ],
        "annotations": [
            {
                "id": "a1",
                "title": "Speed differs",
                "category": "tds",
                "variables": ["/data_01/speed"],
                "figures": [
                    {
                        "variable": "/data_01/speed",
                        "plots": ["Signal Difference"],
                        "zoom": [1, 5],
                    }
                ],
                "future_key": "ignored",
            },
            {"id": "a2", "variables": ["/data_01/*"]},
        ],
    }


def test_load_valid_document(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(_raw()), encoding="utf-8")
    doc = load_annotations(path)
    assert doc.source_label == "PDAP"
    assert doc.annotation("a1").figures[0].zoom == [1.0, 5.0]
    assert [s.id for s, _, _ in doc.iter_sections()] == ["s1", "s1-1", "acc"]
    assert [a.id for a in doc.unplaced_annotations()] == ["a2"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["annotations"].append({"id": "a1"}), "duplicate id 'a1'"),
        (lambda r: r["annotations"][0].update(category="nope"), "unknown category"),
        (lambda r: r["annotations"][0].update(status="bogus"), "status 'bogus'"),
        (
            lambda r: r["annotations"][0]["figures"][0].update(plots=["Pie"]),
            "unknown plot 'Pie'",
        ),
        (
            lambda r: r["annotations"][0]["figures"][0].update(zoom=[5, 1]),
            "xmin < xmax",
        ),
        (
            lambda r: r["annotations"][0]["figures"][0].update(zoom="x"),
            "expected [xmin, xmax]",
        ),
        (lambda r: r["sections"][1].update(auto="magic"), "auto 'magic'"),
        (lambda r: r["sections"][0].update(annotations=["zz"]), "unknown annotation"),
        (lambda r: r.update(schema_version=99), "schema_version"),
        (lambda r: r.update(authors=3), "authors"),
    ],
)
def test_validation_errors(mutate, message: str) -> None:
    raw = _raw()
    mutate(raw)
    with pytest.raises(AnnotationError, match=message.replace("[", r"\[")):
        document_from_dict(raw)


def test_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(AnnotationError, match="invalid JSON"):
        load_annotations(path)


def test_dump_roundtrip_is_stable() -> None:
    doc = document_from_dict(_raw())
    text = dump_annotations(doc)
    assert "future_key" not in text
    assert dump_annotations(document_from_dict(json.loads(text))) == text


def test_glob_matching() -> None:
    doc = document_from_dict(_raw())
    matched = match_annotations(
        doc, ["/data_01/speed", "/data_01/altitude", "/data_02/x"]
    )
    assert [a.id for a in matched["/data_01/speed"]] == ["a1", "a2"]
    assert [a.id for a in matched["/data_01/altitude"]] == ["a2"]
    assert "/data_02/x" not in matched


def test_check_against_results(open_nc_pair) -> None:
    src, tgt = open_nc_pair
    results = scan_all_groups(src, tgt, None)
    raw = _raw()
    raw["annotations"] = [raw["annotations"][0]]
    raw["annotations"][0]["variables"].append("/nowhere/*")
    warnings, unannotated = check_against_results(document_from_dict(raw), results)
    assert any("/nowhere/*" in w for w in warnings)
    assert unannotated == ["/data_01/temperature"]


def test_plot_labels_cover_build_all_figures() -> None:
    import inspect

    from app import plots

    source = inspect.getsource(plots.build_all_figures)
    for label in PLOT_LABELS:
        assert f'"{label}"' in source


def test_edit_helpers() -> None:
    doc = document_from_dict(_raw())
    upsert_annotation(doc, Annotation(id="a1b", title="renamed"), previous_id="a1")
    assert doc.section("s1-1").annotations == ["a1b"]
    upsert_annotation(doc, Annotation(id="a3"), section_id="s1-1")
    assert doc.section("s1-1").annotations == ["a1b", "a3"]
    delete_annotation(doc, "a1b")
    assert doc.section("s1-1").annotations == ["a3"]
    upsert_section(doc, Section(id="s0", title="New"))
    move_section(doc, "s0", -5)
    assert [s.id for s in doc.sections] == ["s0", "s1", "acc"]
    upsert_section(doc, Section(id="s1", title="Renamed"))
    assert doc.section("s1").subsections[0].id == "s1-1"
    assert unique_id("Data 01/Speed", ["data-01-speed"]) == "data-01-speed-2"


def test_template_and_schema(open_nc_pair) -> None:
    src, tgt = open_nc_pair
    doc = template_from_results(scan_all_groups(src, tgt, None))
    ids = [a.variables[0] for a in doc.annotations]
    assert ids == ["/data_01/speed", "/data_01/temperature"]
    document_from_dict(doc.to_dict())
    schema = json_schema()
    assert "annotations" in schema["properties"]
    assert new_document().categories


def test_cli_check(tmp_path: Path, tmp_nc_pair, capsys) -> None:
    src, tgt = tmp_nc_pair
    out = tmp_path / "tpl.json"
    assert main(["template", str(src), str(tgt), "-o", str(out)]) == 0
    assert main(["check", str(out), str(src), str(tgt)]) == 0
    doc = load_annotations(out)
    doc.annotations = [
        Annotation(id="x", variables=["/data_01/speed"], figures=[FigureSpec("/a")])
    ]
    doc.sections = []
    out.write_text(dump_annotations(doc), encoding="utf-8")
    assert main(["check", str(out), str(src), str(tgt)]) == 2
    assert "unannotated: /data_01/temperature" in capsys.readouterr().err
