"""Core comparison logic for NetCDF datasets."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Literal

import numpy

if TYPE_CHECKING:
    import netCDF4

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class FilterInfo:
    """Information about applied filters.

    Attributes:
        expression: The filter expression applied.
        selected_points: Number of points selected by the filter.
        total_points: Total number of points before filtering.
        affected_groups: Set of group names affected by the filter.

    """

    expression: str
    selected_points: int
    total_points: int
    affected_groups: set[str]

    def get_filter_description(self, group_path: str) -> str:
        """Get a description of the filter for a specific group."""
        normalized_path = group_path.lstrip("/")

        is_affected = False
        for affected_group in self.affected_groups:
            if normalized_path == affected_group or normalized_path.startswith(
                affected_group + "/"
            ):
                is_affected = True
                break

        if is_affected:
            masked_points = self.total_points - self.selected_points
            return (
                f"Filter applied: {self.expression} | "
                f"Points masked: {masked_points:,}/{self.total_points:,} "
                f"({100 * masked_points / self.total_points:.1f}%)"
            )
        return ""


@dataclasses.dataclass
class Stats:
    """Statistics of a variable comparison.

    Attributes:
        min_diff: The minimum difference.
        max_diff: The maximum difference.
        mean_diff: The mean difference.
        std_diff: The standard deviation of the difference.

    """

    min_diff: numpy.float64
    max_diff: numpy.float64
    mean_diff: numpy.float64
    std_diff: numpy.float64

    @staticmethod
    def from_array(array: numpy.ndarray) -> Stats:
        """Create a Stats object from a numpy array."""
        return Stats(
            min_diff=numpy.min(array),
            max_diff=numpy.max(array),
            mean_diff=numpy.mean(array).astype(numpy.float64),
            std_diff=numpy.std(array).astype(numpy.float64),
        )

    def __str__(self) -> str:
        """Return a string representation of the statistics."""

        def format_float(x: numpy.float64) -> str:
            rounded = numpy.round(x, 12)
            result = f"{rounded:.6f}"
            if "." in result:
                result = result.rstrip("0")
                if result.endswith("."):
                    result = result[:-1]
            return result

        return (
            f"min={format_float(self.min_diff)}, "
            f"max={format_float(self.max_diff)}, "
            f"mean={format_float(self.mean_diff)}, "
            f"std={format_float(self.std_diff)}"
        )


@dataclasses.dataclass
class VariableComparisonResult:
    """Result of comparing a single variable between source and target.

    Attributes:
        varpath: Full path of the variable (e.g., '/data_01/altitude').
        varname: Short variable name.
        long_name: Human-readable long name.
        status: Comparison status.
        shape_diff: Whether shapes differ.
        dtype_diff: Whether dtypes differ.
        stats: Difference statistics (only for different/almost_equal).
        atol: Absolute tolerance used for comparison.
        message: Optional status message.

    """

    varpath: str
    varname: str
    long_name: str
    status: Literal[
        "equal",
        "almost_equal",
        "different",
        "masked",
        "missing_in_target",
        "missing_in_source",
    ]
    shape_diff: bool = False
    dtype_diff: bool = False
    stats: Stats | None = None
    atol: float = 0.0
    message: str = ""


@dataclasses.dataclass
class AttributeComparisonResult:
    """Result of comparing a single attribute.

    Attributes:
        name: Attribute name.
        src_value: Source value as string.
        tgt_value: Target value as string.

    """

    name: str
    src_value: str
    tgt_value: str


@dataclasses.dataclass
class GroupComparisonResult:
    """Result of comparing a NetCDF group.

    Attributes:
        group_path: Path of the group.
        attributes: List of differing attributes.
        variables: List of variable comparison results.
        filter_description: Optional filter description for this group.

    """

    group_path: str
    attributes: list[AttributeComparisonResult]
    variables: list[VariableComparisonResult]
    filter_description: str = ""


def get_scale_factor(
    src_var: netCDF4.Variable, tgt_var: netCDF4.Variable, varname: str
) -> float:
    """Return the scale factor of the variables, or 0.0 if not present.

    Raise an error if the scale_factor attributes are different.

    Args:
        src_var: The source NetCDF variable.
        tgt_var: The target NetCDF variable.
        varname: The name of the variable.

    Returns:
        The scale factor.

    """

    def check_variable(v: netCDF4.Variable) -> bool:
        val = v[:]
        return "scale_factor" in v.ncattrs() and numpy.issubdtype(
            val.dtype, numpy.floating
        )

    default_tol = 0.0
    tol_ref = (
        src_var.getncattr("scale_factor") if check_variable(src_var) else default_tol
    )
    tol_tgt = (
        tgt_var.getncattr("scale_factor") if check_variable(tgt_var) else default_tol
    )
    if tol_ref != tol_tgt:
        raise ValueError(
            f"Different scale_factor attributes for variable "
            f"{varname}: reference={tol_ref}, target={tol_tgt}."
        )
    return tol_ref


def compare_attributes(
    source: netCDF4.Dataset,
    target: netCDF4.Dataset,
) -> list[AttributeComparisonResult]:
    """Compare attributes of two NetCDF groups.

    Args:
        source: The source NetCDF group.
        target: The target NetCDF group.

    Returns:
        List of differing attributes.

    """
    results: list[AttributeComparisonResult] = []
    for att in source.ncattrs():
        src_att = source.getncattr(att)
        tgt_att = target.getncattr(att) if att in target.ncattrs() else "missing"

        if isinstance(src_att, str) and "," in src_att:
            src_att = {item.strip() for item in src_att.split(",")}
            if tgt_att != "missing":
                tgt_att = {item.strip() for item in tgt_att.split(",")}

        if src_att != tgt_att:
            src_str = ", ".join(src_att) if isinstance(src_att, set) else str(src_att)
            tgt_str = ", ".join(tgt_att) if isinstance(tgt_att, set) else str(tgt_att)
            results.append(
                AttributeComparisonResult(
                    name=att, src_value=src_str, tgt_value=tgt_str
                )
            )
    return results


def _apply_mask(data: numpy.ndarray, mask: numpy.ndarray) -> numpy.ma.MaskedArray:
    """Apply a boolean mask to data, masking points where mask is False."""
    if isinstance(data, numpy.ma.MaskedArray):
        new_mask = numpy.logical_or(data.mask, ~mask)
        return numpy.ma.MaskedArray(data.data, mask=new_mask)
    return numpy.ma.masked_where(~mask, data)


def compare_variable(
    src_var: netCDF4.Variable,
    tgt_var: netCDF4.Variable,
    varpath: str,
    varname: str,
    long_name: str,
    mask: numpy.ndarray | None = None,
) -> VariableComparisonResult:
    """Compare a single variable between source and target.

    Args:
        src_var: Source variable.
        tgt_var: Target variable.
        varpath: Full variable path.
        varname: Short variable name.
        long_name: Human-readable name.
        mask: Optional boolean mask to apply before comparison.

    Returns:
        Comparison result for this variable.

    """
    shape_diff = src_var.shape != tgt_var.shape
    dtype_diff = src_var.dtype != tgt_var.dtype

    src_val = src_var[:]
    tgt_val = tgt_var[:]

    # Apply expression mask if provided (skip if sizes don't match)
    if mask is not None and src_val.shape == mask.shape:
        src_val = _apply_mask(src_val, mask)
        tgt_val = _apply_mask(tgt_val, mask)

    if not isinstance(tgt_val, numpy.ma.MaskedArray):
        tgt_val = numpy.ma.masked_array(tgt_val, mask=False)

    if tgt_val.all() is numpy.ma.masked:
        return VariableComparisonResult(
            varpath=varpath,
            varname=varname,
            long_name=long_name,
            status="masked",
            shape_diff=shape_diff,
            dtype_diff=dtype_diff,
            message="Target values are all masked",
        )

    if src_val.all() is numpy.ma.masked:
        return VariableComparisonResult(
            varpath=varpath,
            varname=varname,
            long_name=long_name,
            status="masked",
            shape_diff=shape_diff,
            dtype_diff=dtype_diff,
            message="Source values are all masked",
        )

    if tgt_val.all() is numpy.ma.masked and src_val.all() is numpy.ma.masked:
        return VariableComparisonResult(
            varpath=varpath,
            varname=varname,
            long_name=long_name,
            status="masked",
            shape_diff=shape_diff,
            dtype_diff=dtype_diff,
            message="Both Target and Source values are all masked",
        )

    if numpy.array_equal(src_val, tgt_val):
        return VariableComparisonResult(
            varpath=varpath,
            varname=varname,
            long_name=long_name,
            status="equal",
            shape_diff=shape_diff,
            dtype_diff=dtype_diff,
        )

    # Values are different - compute stats and check tolerance
    tol = get_scale_factor(src_var, tgt_var, varpath)
    diff = numpy.abs(src_val - tgt_val)
    stats = Stats.from_array(diff)

    if numpy.allclose(src_val, tgt_val, rtol=0, atol=tol, equal_nan=True):
        status = "almost_equal"
    else:
        status = "different"

    return VariableComparisonResult(
        varpath=varpath,
        varname=varname,
        long_name=long_name,
        status=status,
        shape_diff=shape_diff,
        dtype_diff=dtype_diff,
        stats=stats,
        atol=tol,
    )


def compare_group(
    source: netCDF4.Dataset,
    target: netCDF4.Dataset,
    filter_info: FilterInfo | None = None,
    mask: numpy.ndarray | None = None,
) -> GroupComparisonResult:
    """Compare a single group (attributes + variables, non-recursive).

    Args:
        source: Source NetCDF group.
        target: Target NetCDF group.
        filter_info: Optional filter information.
        mask: Optional boolean mask for expression filtering.

    Returns:
        Comparison result for this group.

    """
    group_path = source.path if hasattr(source, "path") else "main"

    # Compare attributes
    attributes = compare_attributes(source, target)

    # Compare variables
    variables: list[VariableComparisonResult] = []
    for varname in sorted(source.variables):
        varobj = source.variables[varname]
        prefix = varobj.group().path
        if prefix != "/":
            prefix += "/"
        varpath = prefix + varname
        long_name = " ".join(
            item.capitalize()
            for item in getattr(
                varobj,
                "long_name",
                varname,
            ).split()
        )

        if varname not in target.variables:
            variables.append(
                VariableComparisonResult(
                    varpath=varpath,
                    varname=varname,
                    long_name=long_name,
                    status="missing_in_target",
                    message="Missing in target",
                )
            )
            continue

        result = compare_variable(
            source.variables[varname],
            target.variables[varname],
            varpath,
            varname,
            long_name,
            mask=mask,
        )
        variables.append(result)

    # Detect variables present in target but absent from source
    for varname in sorted(target.variables):
        if varname in source.variables:
            continue
        varobj = target.variables[varname]
        prefix = varobj.group().path
        if prefix != "/":
            prefix += "/"
        varpath = prefix + varname
        long_name = " ".join(
            item.capitalize() for item in getattr(varobj, "long_name", varname).split()
        )
        variables.append(
            VariableComparisonResult(
                varpath=varpath,
                varname=varname,
                long_name=long_name,
                status="missing_in_source",
                message="Missing in source",
            )
        )

    filter_description = ""
    if filter_info is not None:
        filter_description = filter_info.get_filter_description(group_path)

    return GroupComparisonResult(
        group_path=group_path,
        attributes=attributes,
        variables=variables,
        filter_description=filter_description,
    )


def scan_all_groups(
    source: netCDF4.Dataset,
    target: netCDF4.Dataset,
    filter_info: FilterInfo | None = None,
    mask: numpy.ndarray | None = None,
) -> list[GroupComparisonResult]:
    """Recursively scan all groups and compare them.

    Args:
        source: Source NetCDF dataset.
        target: Target NetCDF dataset.
        filter_info: Optional filter information.
        mask: Optional boolean mask for expression filtering.

    Returns:
        List of group comparison results.

    """
    results: list[GroupComparisonResult] = []

    def _scan(src_grp: netCDF4.Dataset, tgt_grp: netCDF4.Dataset) -> None:
        LOGGER.info("Comparing group: %s", src_grp.path)
        result = compare_group(src_grp, tgt_grp, filter_info, mask=mask)
        results.append(result)
        for grp_name in src_grp.groups:
            if grp_name in tgt_grp.groups:
                _scan(src_grp.groups[grp_name], tgt_grp.groups[grp_name])

    _scan(source, target)
    return results


def rescan_affected_groups(
    source: netCDF4.Dataset,
    target: netCDF4.Dataset,
    previous_results: list[GroupComparisonResult],
    filter_info: FilterInfo | None = None,
    mask: numpy.ndarray | None = None,
) -> list[GroupComparisonResult]:
    """Re-scan only groups affected by the filter, reuse others.

    When a filter is applied, only groups whose variables match the mask
    shape need to be recomputed. Groups unaffected by the filter keep
    their previous comparison results (with updated filter_description).

    Args:
        source: Source NetCDF dataset.
        target: Target NetCDF dataset.
        previous_results: Results from a prior scan_all_groups call.
        filter_info: Optional filter information.
        mask: Optional boolean mask for expression filtering.

    Returns:
        Updated list of group comparison results.

    """
    if mask is None or not previous_results:
        return scan_all_groups(source, target, filter_info, mask=mask)

    affected_groups: set[str] = set()
    if filter_info is not None:
        affected_groups = filter_info.affected_groups

    # Build a lookup of previous results by group_path
    prev_by_path = {r.group_path: r for r in previous_results}

    results: list[GroupComparisonResult] = []

    def _rescan(src_grp: netCDF4.Dataset, tgt_grp: netCDF4.Dataset) -> None:
        group_path = src_grp.path if hasattr(src_grp, "path") else "main"
        normalized = group_path.lstrip("/")

        # Check if this group is affected by the filter
        is_affected = any(
            normalized == ag or normalized.startswith(ag + "/")
            for ag in affected_groups
        )

        if is_affected:
            LOGGER.info("Recomparing affected group: %s", group_path)
            result = compare_group(src_grp, tgt_grp, filter_info, mask=mask)
        elif group_path in prev_by_path:
            LOGGER.info("Reusing previous result for group: %s", group_path)
            prev = prev_by_path[group_path]
            # Update filter_description even for unaffected groups
            desc = ""
            if filter_info is not None:
                desc = filter_info.get_filter_description(group_path)
            result = GroupComparisonResult(
                group_path=prev.group_path,
                attributes=prev.attributes,
                variables=prev.variables,
                filter_description=desc,
            )
        else:
            LOGGER.info("Comparing new group: %s", group_path)
            result = compare_group(src_grp, tgt_grp, filter_info, mask=mask)

        results.append(result)
        for grp_name in src_grp.groups:
            if grp_name in tgt_grp.groups:
                _rescan(src_grp.groups[grp_name], tgt_grp.groups[grp_name])

    _rescan(source, target)
    return results
