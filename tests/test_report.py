"""Tests for the CSV / JSON report exporter."""

from __future__ import annotations

import csv
import io
import json

import numpy

from app.comparison import (
    AttributeComparisonResult,
    GroupComparisonResult,
    Stats,
    VariableComparisonResult,
)
from app.report import results_to_csv, results_to_json


def _fixture_results() -> list[GroupComparisonResult]:
    stats = Stats.from_array(numpy.array([0.1, 0.2, 0.3]))
    return [
        GroupComparisonResult(
            group_path="/data_01",
            attributes=[
                AttributeComparisonResult(name="title", src_value="a", tgt_value="b")
            ],
            variables=[
                VariableComparisonResult(
                    varpath="/data_01/altitude",
                    varname="altitude",
                    long_name="Altitude",
                    status="equal",
                ),
                VariableComparisonResult(
                    varpath="/data_01/speed",
                    varname="speed",
                    long_name="Speed",
                    status="different",
                    stats=stats,
                    atol=0.01,
                ),
            ],
        )
    ]


def test_csv_has_row_per_variable():
    csv_text = results_to_csv(_fixture_results())
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["varname"] == "altitude"
    assert rows[0]["status"] == "equal"
    assert rows[0]["mean_diff"] == ""
    assert rows[1]["varname"] == "speed"
    assert rows[1]["status"] == "different"
    assert rows[1]["mean_diff"] != ""


def test_json_roundtrips_as_list_of_groups():
    payload = json.loads(results_to_json(_fixture_results()))
    assert isinstance(payload, list) and len(payload) == 1
    group = payload[0]
    assert group["group_path"] == "/data_01"
    assert len(group["variables"]) == 2
    assert group["variables"][1]["stats"]["mean_diff"] > 0
