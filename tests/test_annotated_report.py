"""Tests for the annotated HTML report and its command-line front-end."""

from __future__ import annotations

import base64
import datetime
import json
import re
from pathlib import Path

import numpy
import pytest

from app.annotated_report import ReportInputs, build_report, figure_to_json
from app.annotations import document_from_dict, dump_annotations
from app.comparison import scan_all_groups
from app.report_cli import main


def _doc(**overrides):
    raw = {
        "language": "fr",
        "title": "Validation <script>alert(1)</script>",
        "source_label": "PDAP",
        "target_label": "c3po",
        "datasets": [{"label": "Passe", "value": "cycle 188"}],
        "categories": [{"id": "tds", "label": "TDS", "color": "#c62828"}],
        "sections": [
            {"id": "objet", "title": "Objet", "body": "Un **texte** <b>brut</b>"},
            {
                "id": "s3",
                "title": "Points",
                "subsections": [
                    {"id": "s3-1", "title": "Vitesse", "annotations": ["speed"]}
                ],
            },
            {"id": "accord", "title": "Accord", "auto": "agreement_table"},
            {"id": "resume", "title": "Résumé", "auto": "summary_counts"},
        ],
        "annotations": [
            {
                "id": "speed",
                "title": "La vitesse diffère",
                "category": "tds",
                "variables": ["/data_01/speed"],
                "text": "| a | b |\n|---|---|\n| 1 | 2 |",
                "effect": "Aucun",
                "figures": [
                    {
                        "variable": "/data_01/speed",
                        "plots": ["Signal Comparison", "Statistics"],
                        "zoom": [2, 5],
                    },
                    {
                        "variable": "/data_01/speed",
                        "plots": ["Histogram"],
                        "expression": "${/data_01/nope} > 1",
                    },
                ],
            },
            {"id": "loose", "title": "Non rangée", "variables": ["/data_01/x*"]},
        ],
    }
    raw.update(overrides)
    return document_from_dict(raw)


def _inputs(open_nc_pair) -> ReportInputs:
    src, tgt = open_nc_pair
    return ReportInputs(
        src_ds=src,
        tgt_ds=tgt,
        src_name="ref.nc",
        tgt_name="tgt.nc",
        results=scan_all_groups(src, tgt, None),
    )


def _array(value) -> list:
    """Decode plotly's typed-array encoding (``{"dtype", "bdata"}``)."""
    if isinstance(value, dict) and "bdata" in value:
        raw = base64.b64decode(value["bdata"])
        return numpy.frombuffer(raw, dtype=value["dtype"]).tolist()
    return list(value)


def _figures(page: str) -> list[dict]:
    blobs = re.findall(
        r'<script type="application/json" id="fig-\d+">(.*?)</script>', page, re.S
    )
    return [json.loads(b) for b in blobs]


def test_report_structure(open_nc_pair) -> None:
    page = build_report(
        _doc(),
        _inputs(open_nc_pair),
        plotlyjs="cdn",
        generated_at=datetime.datetime(2026, 10, 12, 9, 30),
    )
    assert page.startswith("<!DOCTYPE html>")
    assert "<script>alert(1)</script>" not in page
    assert "Validation &lt;script&gt;" in page
    assert "<strong>texte</strong>" in page
    assert "<b>brut</b>" not in page
    assert 'id="sec-s3-1"' in page and "2.1. Vitesse" in page
    assert "<table>" in page
    assert "La vitesse diffère" in page
    assert "Autres annotations" in page and "Non rangée" in page
    assert "/data_01/temperature" in page and "non annoté" in page
    assert "Expression invalide" in page
    assert "2026-10-12 09:30" in page
    assert "cdn.plot.ly" in page


def test_report_figures_are_zoomed_and_relabelled(open_nc_pair) -> None:
    page = build_report(_doc(), _inputs(open_nc_pair), plotlyjs="cdn")
    figs = _figures(page)
    assert len(figs) == 2
    signal = figs[0]
    assert signal["layout"]["xaxis"]["range"] == [2.0, 5.0]
    assert {t["name"] for t in signal["data"]} >= {"PDAP", "c3po"}
    assert _array(signal["data"][0]["x"]) == [2, 3, 4, 5]
    stats = figs[1]["data"][0]["cells"]["values"][1]
    assert stats[0] == "4"


def test_report_inline_plotly_and_no_figures(open_nc_pair) -> None:
    inputs = _inputs(open_nc_pair)
    page = build_report(_doc(), inputs)
    assert '<script src="https://cdn.plot.ly' not in page
    assert len(page) > 1_000_000
    empty = build_report(_doc(annotations=[], sections=[]), inputs)
    assert "<script src=" not in empty and len(empty) < 100_000


def test_figure_to_json_escapes_script_end() -> None:
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=[1, float("nan")], y=[1, 2], name="</script>"))
    text = figure_to_json(fig)
    assert "</script>" not in text
    assert _array(json.loads(text)["data"][0]["y"]) == [1, 2]


def test_bad_plotlyjs_mode(open_nc_pair) -> None:
    with pytest.raises(ValueError):
        build_report(_doc(), _inputs(open_nc_pair), plotlyjs="bogus")


def test_cli(tmp_path: Path, tmp_nc_pair, capsys) -> None:
    src, tgt = tmp_nc_pair
    ann = tmp_path / "ann.json"
    ann.write_text(dump_annotations(_doc()), encoding="utf-8")
    out = tmp_path / "r.html"
    args = [str(src), str(tgt), "-a", str(ann), "-o", str(out), "--plotlyjs", "cdn"]
    assert main(args) == 0
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    err = capsys.readouterr().err
    assert "no annotation for /data_01/temperature" in err
    assert "'/data_01/x*' matches no variable" in err
    assert main([*args, "--strict"]) == 2
    assert main([*args, "--expr", "${/data_01/flag} == 0"]) == 0
    assert main([*args, "--expr", "${/nope} > 1"]) == 1


def test_cli_rejects_invalid_annotations(tmp_path: Path, tmp_nc_pair) -> None:
    src, tgt = tmp_nc_pair
    ann = tmp_path / "ann.json"
    ann.write_text('{"annotations": [{"id": "a", "status": "bad"}]}')
    assert main([str(src), str(tgt), "-a", str(ann), "-o", str(tmp_path / "x")]) == 1


def test_pdf(tmp_path: Path, tmp_nc_pair) -> None:
    pytest.importorskip("playwright")
    src, tgt = tmp_nc_pair
    ann = tmp_path / "ann.json"
    ann.write_text(dump_annotations(_doc()), encoding="utf-8")
    out = tmp_path / "r.pdf"
    assert main([str(src), str(tgt), "-a", str(ann), "-o", str(out)]) == 0
    assert out.read_bytes().startswith(b"%PDF")
