"""Tests for the data_loading module."""

from __future__ import annotations

import numpy

from app.data_loading import (
    DataSource,
    _promote_integer_to_float,
    lttb_downsample,
)

# ── _promote_integer_to_float ─────────────────────────────────────────


class TestPromoteIntegerToFloat:
    def test_int16_to_float32(self):
        arr = numpy.array([1, 2, 3], dtype=numpy.int16)
        result = _promote_integer_to_float(arr)
        assert result.dtype == numpy.float32

    def test_int32_to_float64(self):
        arr = numpy.array([1, 2, 3], dtype=numpy.int32)
        result = _promote_integer_to_float(arr)
        assert result.dtype == numpy.float64

    def test_int64_to_float64(self):
        arr = numpy.array([1, 2, 3], dtype=numpy.int64)
        result = _promote_integer_to_float(arr)
        assert result.dtype == numpy.float64


# ── lttb_downsample ───────────────────────────────────────────────────


class TestLttbDownsample:
    def test_no_downsample_when_small(self):
        x = numpy.arange(10, dtype=float)
        y = numpy.sin(x)
        rx, ry = lttb_downsample(x, y, 20)
        numpy.testing.assert_array_equal(rx, x)
        numpy.testing.assert_array_equal(ry, y)

    def test_downsample_reduces_length(self):
        x = numpy.arange(1000, dtype=float)
        y = numpy.sin(x / 100.0)
        rx, ry = lttb_downsample(x, y, 100)
        assert len(rx) == 100
        assert len(ry) == 100

    def test_first_and_last_preserved(self):
        x = numpy.arange(500, dtype=float)
        y = numpy.random.default_rng(42).random(500)
        rx, ry = lttb_downsample(x, y, 50)
        assert rx[0] == x[0]
        assert rx[-1] == x[-1]
        assert ry[0] == y[0]
        assert ry[-1] == y[-1]

    def test_handles_nan(self):
        x = numpy.arange(200, dtype=float)
        y = numpy.ones(200)
        y[50:60] = numpy.nan
        rx, _ry = lttb_downsample(x, y, 50)
        assert len(rx) == 50
        # Should not raise or produce all NaN


# ── DataSource ────────────────────────────────────────────────────────


class TestDataSource:
    def test_basic_construction(self):
        expected = numpy.array([1.0, 2.0, 3.0])
        computed = numpy.array([1.1, 2.0, 3.2])
        ds = DataSource(expected, computed, flag=None)
        assert "x" in ds.source_data
        assert "expected" in ds.source_data
        assert "computed" in ds.source_data
        assert "difference" in ds.source_data
        assert len(ds.source_data["x"]) == 3

    def test_difference_computed(self):
        expected = numpy.array([1.0, 2.0, 3.0])
        computed = numpy.array([1.5, 2.5, 3.5])
        ds = DataSource(expected, computed, flag=None)
        numpy.testing.assert_array_almost_equal(
            ds.source_data["difference"], [0.5, 0.5, 0.5]
        )

    def test_histogram_data(self):
        expected = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        computed = numpy.array([1.1, 2.2, 3.3, 4.4, 5.5])
        ds = DataSource(expected, computed, flag=None)
        assert "top" in ds.histogram_data
        assert "left" in ds.histogram_data
        assert "right" in ds.histogram_data

    def test_stats_data(self):
        expected = numpy.array([1.0, 2.0, 3.0])
        computed = numpy.array([1.5, 2.5, 3.5])
        ds = DataSource(expected, computed, flag=None)
        assert "metric" in ds.stats_data
        assert "value" in ds.stats_data
        assert ds.stats_data["value"][0] == 3  # Samples

    def test_nan_distribution(self):
        expected = numpy.array([1.0, numpy.nan, 3.0])
        computed = numpy.array([1.0, 2.0, numpy.nan])
        ds = DataSource(expected, computed, flag=None)
        assert ds.exclusive_nan

    def test_no_exclusive_nan(self):
        expected = numpy.array([1.0, 2.0, 3.0])
        computed = numpy.array([1.0, 2.0, 3.0])
        ds = DataSource(expected, computed, flag=None)
        assert not ds.exclusive_nan

    def test_integer_data(self):
        expected = numpy.array([1, 2, 3], dtype=numpy.int32)
        computed = numpy.array([1, 2, 4], dtype=numpy.int32)
        ds = DataSource(expected, computed, flag=None)
        # Should be promoted to float
        assert ds.full_expected.dtype in (numpy.float32, numpy.float64)

    def test_with_coordinates(self):
        expected = numpy.array([1.0, 2.0, 3.0])
        computed = numpy.array([1.0, 2.0, 3.0])
        lon = numpy.array([0.0, 1.0, 2.0])
        lat = numpy.array([45.0, 46.0, 47.0])
        ds = DataSource(expected, computed, flag=None, longitude=lon, latitude=lat)
        assert "longitude" in ds.source_data
        assert "latitude" in ds.source_data

    def test_downsampling_large_data(self):
        n = 50_000
        expected = numpy.random.default_rng(0).random(n).astype(numpy.float32)
        computed = expected + 0.01
        ds = DataSource(expected, computed, flag=None)
        # Should be downsampled
        assert len(ds.source_data["x"]) <= 20_000

    def test_apply_flag_filter(self):
        expected = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        computed = numpy.array([1.5, 2.5, 3.5, 4.5, 5.5])
        ds = DataSource(expected, computed, flag=None)
        # Filter to keep only indices 0, 2, 4
        indices = numpy.array([0, 2, 4])
        ds.apply_flag_filter(indices)
        assert len(ds.source_data["x"]) == 3
        numpy.testing.assert_array_equal(ds.source_data["x"], [0, 2, 4])

    def test_get_zoomed_source_data_returns_full_in_window(self):
        """Zooming into a small window should return all points in that range."""
        expected = numpy.arange(100, dtype=float)
        computed = numpy.arange(100, dtype=float) + 0.5
        ds = DataSource(expected, computed, flag=None)
        zd = ds.get_zoomed_source_data(10.0, 20.0)
        # Should contain exactly indices 10..20
        assert zd["x"][0] == 10.0
        assert zd["x"][-1] == 20.0
        assert len(zd["x"]) == 11
        assert "expected" in zd
        assert "computed" in zd
        assert "difference" in zd

    def test_get_zoomed_source_data_downsamples_large(self):
        """With >MAX_DISPLAY_POINTS in zoom window, LTTB should downsample."""
        n = 50_000
        expected = numpy.random.default_rng(42).random(n)
        computed = numpy.random.default_rng(43).random(n)
        ds = DataSource(expected, computed, flag=None)
        # Zoom on full range — should still downsample
        zd = ds.get_zoomed_source_data(0.0, float(n - 1))
        assert len(zd["x"]) <= 20_000

    def test_get_zoomed_source_data_reset_on_full(self):
        """source_data should represent the global downsampled view."""
        expected = numpy.arange(50, dtype=float)
        computed = numpy.arange(50, dtype=float) * 2
        ds = DataSource(expected, computed, flag=None)
        # source_data covers everything
        assert len(ds.source_data["x"]) == 50
        # Zoom in then verify original is untouched
        zd = ds.get_zoomed_source_data(5.0, 10.0)
        assert len(zd["x"]) == 6
        assert len(ds.source_data["x"]) == 50  # unchanged
