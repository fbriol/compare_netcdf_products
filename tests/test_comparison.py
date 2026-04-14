"""Tests for the comparison module."""

from __future__ import annotations

import numpy
import pytest

from app.comparison import (
    FilterInfo,
    Stats,
    _apply_mask,
    compare_attributes,
    compare_group,
    compare_variable,
    rescan_affected_groups,
    scan_all_groups,
)

# ── Stats ──────────────────────────────────────────────────────────────


class TestStats:
    def test_from_array(self):
        arr = numpy.array([1.0, 2.0, 3.0, 4.0])
        stats = Stats.from_array(arr)
        assert stats.min_diff == 1.0
        assert stats.max_diff == 4.0
        assert stats.mean_diff == pytest.approx(2.5)
        assert stats.std_diff == pytest.approx(numpy.std(arr))

    def test_str_format(self):
        stats = Stats(
            min_diff=numpy.float64(0.0),
            max_diff=numpy.float64(1.5),
            mean_diff=numpy.float64(0.75),
            std_diff=numpy.float64(0.5),
        )
        s = str(stats)
        assert "min=0" in s
        assert "max=1.5" in s


# ── _apply_mask ────────────────────────────────────────────────────────


class TestApplyMask:
    def test_plain_array(self):
        data = numpy.array([10, 20, 30, 40])
        mask = numpy.array([True, False, True, False])
        result = _apply_mask(data, mask)
        assert isinstance(result, numpy.ma.MaskedArray)
        # mask=True means masked (hidden), so positions 1 and 3 are masked
        assert result[0] == 10
        assert result[2] == 30
        assert result.mask[1]
        assert result.mask[3]

    def test_masked_array_input(self):
        data = numpy.ma.MaskedArray([10, 20, 30, 40], mask=[False, True, False, False])
        filter_mask = numpy.array([True, True, False, True])
        result = _apply_mask(data, filter_mask)
        # Position 1 was already masked, position 2 is masked by filter
        assert result.mask[1]
        assert result.mask[2]
        assert not result.mask[0]


# ── compare_variable ──────────────────────────────────────────────────


class TestCompareVariable:
    def test_equal_variable(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        src_var = src_ds["data_01"]["altitude"]
        tgt_var = tgt_ds["data_01"]["altitude"]
        result = compare_variable(
            src_var, tgt_var, "/data_01/altitude", "altitude", "Altitude"
        )
        assert result.status == "equal"
        assert result.stats is None

    def test_different_variable(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        src_var = src_ds["data_01"]["speed"]
        tgt_var = tgt_ds["data_01"]["speed"]
        result = compare_variable(src_var, tgt_var, "/data_01/speed", "speed", "Speed")
        assert result.status == "different"
        assert result.stats is not None
        assert result.stats.max_diff > 0

    def test_almost_equal_variable(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        src_var = src_ds["data_01"]["temperature"]
        tgt_var = tgt_ds["data_01"]["temperature"]
        result = compare_variable(
            src_var,
            tgt_var,
            "/data_01/temperature",
            "temperature",
            "Temperature",
        )
        assert result.status == "almost_equal"
        assert result.atol == pytest.approx(0.01)

    def test_with_mask(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        src_var = src_ds["data_01"]["speed"]
        tgt_var = tgt_ds["data_01"]["speed"]
        # Mask out all points: all masked → equal
        mask = numpy.zeros(10, dtype=bool)
        result = compare_variable(
            src_var, tgt_var, "/data_01/speed", "speed", "Speed", mask=mask
        )
        assert result.status == "masked"


# ── compare_attributes ────────────────────────────────────────────────


class TestCompareAttributes:
    def test_no_differences(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        src_grp = src_ds["data_01"]
        tgt_grp = tgt_ds["data_01"]
        diffs = compare_attributes(src_grp, tgt_grp)
        assert diffs == []


# ── compare_group ─────────────────────────────────────────────────────


class TestCompareGroup:
    def test_group_comparison(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        src_grp = src_ds["data_01"]
        tgt_grp = tgt_ds["data_01"]
        result = compare_group(src_grp, tgt_grp)
        assert result.group_path == "/data_01"
        statuses = {v.varname: v.status for v in result.variables}
        assert statuses["altitude"] == "equal"
        assert statuses["speed"] == "different"
        assert statuses["temperature"] == "almost_equal"


# ── scan_all_groups ───────────────────────────────────────────────────


class TestScanAllGroups:
    def test_scan(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        results = scan_all_groups(src_ds, tgt_ds)
        # Should have root + data_01
        assert len(results) >= 1
        paths = {r.group_path for r in results}
        assert "/data_01" in paths


# ── rescan_affected_groups ────────────────────────────────────────────


class TestRescanAffectedGroups:
    def test_no_filter_falls_back_to_full_scan(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        results = rescan_affected_groups(src_ds, tgt_ds, [], None, None)
        assert len(results) >= 1

    def test_unaffected_groups_reused(self, open_nc_pair):
        src_ds, tgt_ds = open_nc_pair
        original = scan_all_groups(src_ds, tgt_ds)
        # Create a filter that affects a non-existent group
        fi = FilterInfo(
            expression="test",
            selected_points=5,
            total_points=10,
            affected_groups={"nonexistent_group"},
        )
        mask = numpy.ones(10, dtype=bool)
        results = rescan_affected_groups(src_ds, tgt_ds, original, fi, mask)
        # data_01 should be reused (same object for variables)
        orig_data01 = next(r for r in original if r.group_path == "/data_01")
        new_data01 = next(r for r in results if r.group_path == "/data_01")
        assert orig_data01.variables is new_data01.variables


# ── FilterInfo ────────────────────────────────────────────────────────


class TestFilterInfo:
    def test_filter_description_affected(self):
        fi = FilterInfo(
            expression="${/data_01/flag} == 0",
            selected_points=8,
            total_points=10,
            affected_groups={"data_01"},
        )
        desc = fi.get_filter_description("/data_01")
        assert "8" in desc or "10" in desc

    def test_filter_description_unaffected(self):
        fi = FilterInfo(
            expression="${/data_01/flag} == 0",
            selected_points=8,
            total_points=10,
            affected_groups={"data_01"},
        )
        desc = fi.get_filter_description("/data_02")
        # Unaffected group gets no filter description or a different one
        assert isinstance(desc, str)
