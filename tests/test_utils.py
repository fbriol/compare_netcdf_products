"""Tests for the utils module."""

from __future__ import annotations

import dataclasses

from app.comparison import GroupComparisonResult, VariableComparisonResult
from app.utils import (
    count_diffs,
    count_diffs_recursive,
    find_group_result,
    navigate_to_group,
)

# ── Minimal state stub ────────────────────────────────────────────────


@dataclasses.dataclass
class _FakeState:
    group_results: list[GroupComparisonResult]


def _make_group_result(group_path, statuses):
    """Create a GroupComparisonResult with variables of given statuses."""
    variables = [
        VariableComparisonResult(
            varpath=f"{group_path}/var{i}",
            varname=f"var{i}",
            long_name=f"Var {i}",
            status=s,
        )
        for i, s in enumerate(statuses)
    ]
    return GroupComparisonResult(
        group_path=group_path, attributes=[], variables=variables
    )


# ── find_group_result ─────────────────────────────────────────────────


class TestFindGroupResult:
    def test_found(self):
        gr = _make_group_result("/data_01", ["equal"])
        state = _FakeState(group_results=[gr])
        assert find_group_result(state, "/data_01") is gr

    def test_not_found(self):
        state = _FakeState(group_results=[])
        assert find_group_result(state, "/nope") is None


# ── navigate_to_group ─────────────────────────────────────────────────


class TestNavigateToGroup:
    def test_root(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        result = navigate_to_group(src_ds, "/")
        assert result is src_ds

    def test_existing_group(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        result = navigate_to_group(src_ds, "/data_01")
        assert result is not None

    def test_nonexistent_group(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        result = navigate_to_group(src_ds, "/nonexistent")
        assert result is None


# ── count_diffs ───────────────────────────────────────────────────────


class TestCountDiffs:
    def test_counts(self):
        gr = _make_group_result(
            "/data_01", ["equal", "different", "almost_equal", "equal"]
        )
        state = _FakeState(group_results=[gr])
        n_diff, n_total = count_diffs(state, "/data_01")
        assert n_diff == 2
        assert n_total == 4

    def test_missing_group(self):
        state = _FakeState(group_results=[])
        assert count_diffs(state, "/nope") == (0, 0)


# ── count_diffs_recursive ────────────────────────────────────────────


class TestCountDiffsRecursive:
    def test_recursive(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        gr_root = _make_group_result("/", ["equal"])
        gr_child = _make_group_result("/data_01", ["different", "equal"])
        state = _FakeState(group_results=[gr_root, gr_child])
        n_diff, n_total = count_diffs_recursive(state, src_ds, "/")
        assert n_diff == 1
        assert n_total == 3
