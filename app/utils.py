"""Shared utility helpers for the application."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import netCDF4

    from .comparison import GroupComparisonResult
    from .state import SessionState


def find_group_result(
    state: SessionState,
    group_path: str,
) -> GroupComparisonResult | None:
    """Find the comparison result for *group_path*.

    Args:
        state: The session state.
        group_path: The group path to look up.

    Returns:
        The matching result, or ``None``.

    """
    for gr in state.group_results:
        if gr.group_path == group_path:
            return gr
    return None


def navigate_to_group(
    dataset: netCDF4.Dataset,
    group_path: str,
) -> netCDF4.Group | None:
    """Navigate to a NetCDF group by path.

    Args:
        dataset: The NetCDF dataset.
        group_path: Slash-separated path (e.g. ``"/data_01"``).

    Returns:
        The group, or ``None`` if the path doesn't exist.

    """
    if group_path == "/":
        return dataset
    parts = group_path.strip("/").split("/")
    group = dataset
    for part in parts:
        if part in group.groups:
            group = group.groups[part]
        else:
            return None
    return group


def count_diffs(state: SessionState, group_path: str) -> tuple[int, int]:
    """Return ``(n_different, n_total)`` variables in *group_path*.

    Args:
        state: The session state.
        group_path: The group path.

    Returns:
        Tuple of different/total variable counts.

    """
    gr = find_group_result(state, group_path)
    if gr is None:
        return 0, 0
    n_diff = sum(1 for v in gr.variables if v.status in ("different", "almost_equal"))
    return n_diff, len(gr.variables)


def count_diffs_recursive(
    state: SessionState,
    dataset: netCDF4.Dataset,
    group_path: str,
) -> tuple[int, int]:
    """Return ``(n_different, n_total)`` across *group_path* and children.

    Args:
        state: The session state.
        dataset: The NetCDF dataset to walk.
        group_path: The root group path.

    Returns:
        Tuple of different/total variable counts (recursive).

    """
    n_diff, n_total = count_diffs(state, group_path)
    group = navigate_to_group(dataset, group_path)
    if group is not None:
        for grp_name in group.groups:
            child_path = (
                f"{group_path}/{grp_name}" if group_path != "/" else f"/{grp_name}"
            )
            cd, ct = count_diffs_recursive(state, dataset, child_path)
            n_diff += cd
            n_total += ct
    return n_diff, n_total
