"""Session state management for the Dash application."""

from __future__ import annotations

import collections
import dataclasses
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    import netCDF4
    import numpy

    from .comparison import FilterInfo, GroupComparisonResult
    from .data_loading import DataSource
    from .flags import Flag

from .constants import MAX_VARIABLE_CACHE_SIZE

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class SessionState:
    """Holds all session-scoped state for the application.

    Attributes:
        src_path: Path to the reference NetCDF file.
        tgt_path: Path to the target NetCDF file.
        src_ds: Open reference NetCDF dataset.
        tgt_ds: Open target NetCDF dataset.
        flags: Optional list of Flag objects for interactive filtering.
        active_mask: Current boolean mask from expression filter (None = no filter).
        active_expression: Current expression string (None = no filter).
        filter_info: Current FilterInfo from expression evaluation.
        group_results: Cached comparison results per group.
        variable_cache: LRU cache for loaded variable DataSources.
        max_cache_size: Maximum number of entries in the variable cache.

    """

    src_path: pathlib.Path
    tgt_path: pathlib.Path
    src_ds: netCDF4.Dataset
    tgt_ds: netCDF4.Dataset
    flags: list[Flag] | None = None
    active_mask: numpy.ndarray | None = None
    active_expression: str | None = None
    filter_info: FilterInfo | None = None
    group_results: list[GroupComparisonResult] = dataclasses.field(default_factory=list)
    variable_cache: collections.OrderedDict[str, DataSource] = dataclasses.field(
        default_factory=collections.OrderedDict
    )
    max_cache_size: int = MAX_VARIABLE_CACHE_SIZE

    # Navigation state
    current_view: str = "group"
    current_group_path: str = "/"
    current_varpath: str | None = None

    def clear_variable_cache(self) -> None:
        """Clear the variable data cache (e.g., after expression change)."""
        self.variable_cache.clear()
        LOGGER.info("Variable cache cleared")

    def close(self) -> None:
        """Close the NetCDF datasets."""
        self.src_ds.close()
        self.tgt_ds.close()
        LOGGER.info("NetCDF datasets closed")
