"""Flag variable handling for NetCDF data filtering."""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from collections.abc import Iterable

    import netCDF4


class Flag:
    """Class to store flag values and their meanings.

    Args:
        varname: The variable path (e.g., '/data_20/wvf_main_class').
        target: The target NetCDF dataset.

    """

    def __init__(
        self,
        varname: str,
        target: netCDF4.Dataset,
    ) -> None:
        """Initialize the Flag object."""
        parts = varname.split("/")
        if not parts[0]:
            parts = parts[1:]
        self.varname = parts.pop()
        self.group = "/".join(parts)

        # Dive into the group to get the variable
        group = target
        for part in parts:
            group = group.groups[part]
        variable = group.variables[self.varname]

        # Extract the flag values and their meanings
        label: list[str] = variable.getncattr("flag_meanings").split()
        flag_values: Iterable[float] = variable.getncattr("flag_values").tolist()

        # Build a mapping from flag values to their meanings
        mapping = dict(zip(flag_values, label, strict=False))

        # Extract the flag values
        self.values: numpy.ndarray = numpy.ma.filled(variable[:])

        # Build a mapping from the variable name/flag meaning to the flag value
        self.mapping = {
            f"{self.varname}/{value}": item for item, value in mapping.items()
        }

        # Build the options for selection widgets
        self.options = [f"{self.varname}/{item}" for item in label]

    def is_in_current_group(self, varpath: str) -> bool:
        """Check if the variable is in the current group.

        Args:
            varpath: The variable path.

        Returns:
            True if the variable is in the current group, False otherwise.

        """
        group = posixpath.dirname(varpath)
        if group[0] == "/":
            group = group[1:]
        return group.startswith(self.group)

    def compute_filtered_indices(
        self,
        selected_options: list[str],
        all_flags: list[Flag],
    ) -> numpy.ndarray:
        """Compute the indices of data points matching the selected flags.

        Args:
            selected_options: List of selected flag options
                (e.g., ['varname/meaning1', 'varname/meaning2']).
            all_flags: List of all Flag objects for building the full filter.

        Returns:
            Array of indices that match the selected flags.

        """
        # Build flag map: {flag_varname: set_of_selected_values}
        flag_map: dict[str, set[float]] = {}
        all_mappings = {
            key: value for flag in all_flags for key, value in flag.mapping.items()
        }
        for item in selected_options:
            flag_name, _ = item.split("/", 1)
            if flag_name not in flag_map:
                flag_map[flag_name] = set()
            flag_map[flag_name].add(all_mappings[item])

        # Find indices that match all flag conditions
        n = len(self.values)
        mask = numpy.ones(n, dtype=bool)
        for flag in all_flags:
            if flag.varname in flag_map:
                allowed = flag_map[flag.varname]
                flag_mask = numpy.isin(flag.values, list(allowed))
                mask &= flag_mask

        return numpy.where(mask)[0]
