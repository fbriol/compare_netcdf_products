"""On-demand data loading and DataSource for Plotly plots."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    import netCDF4

    from .flags import Flag
    from .state import SessionState

from .comparison import _apply_mask
from .constants import MAX_DISPLAY_POINTS

LOGGER = logging.getLogger(__name__)


def _promote_integer_to_float(data: numpy.ndarray) -> numpy.ndarray:
    """Promotes integer data to float.

    Args:
        data: The data to promote.

    Returns:
        The promoted data.

    """
    assert issubclass(data.dtype.type, numpy.integer)
    if data.dtype.itemsize <= 2:
        return data.astype(numpy.float32)
    return data.astype(numpy.float64)


def lttb_downsample(
    x: numpy.ndarray, y: numpy.ndarray, n_out: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Downsample using Largest-Triangle-Three-Bucket algorithm.

    Vectorised per-bucket selection for speed.

    Args:
        x: X values.
        y: Y values.
        n_out: Target number of output points.

    Returns:
        Tuple of (downsampled_x, downsampled_y).

    """
    n = len(x)
    if n <= n_out:
        return x, y

    # Replace NaN in y with 0 for area computation (avoids NaN propagation)
    y_safe = numpy.where(numpy.isnan(y), 0.0, y)

    sampled_indices = numpy.empty(n_out, dtype=numpy.intp)
    sampled_indices[0] = 0
    sampled_indices[-1] = n - 1

    bucket_size = (n - 2) / (n_out - 2)
    a = 0  # index of previously selected point

    for i in range(1, n_out - 1):
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = min(int(i * bucket_size) + 1, n)

        next_start = int(i * bucket_size) + 1
        next_end = min(int((i + 1) * bucket_size) + 1, n)

        avg_x = numpy.mean(x[next_start:next_end])
        avg_y = numpy.mean(y_safe[next_start:next_end])

        # Vectorised area computation for the whole bucket
        bx = x[bucket_start:bucket_end]
        by = y_safe[bucket_start:bucket_end]
        areas = numpy.abs(
            (x[a] - avg_x) * (by - y_safe[a]) - (x[a] - bx) * (avg_y - y_safe[a])
        )
        best = int(numpy.argmax(areas)) + bucket_start
        sampled_indices[i] = best
        a = best

    return x[sampled_indices], y[sampled_indices]


class DataSource:
    """Data source for plots, encapsulating variable data and statistics.

    Args:
        expected: The expected (reference) values.
        computed: The computed (target) values.
        flag: Optional list of flags for filtering.

    """

    def __init__(
        self,
        expected: numpy.ndarray,
        computed: numpy.ndarray,
        flag: list[Flag] | None,
        longitude: numpy.ndarray | None = None,
        latitude: numpy.ndarray | None = None,
    ) -> None:
        """Initialize the data source."""
        self.full_longitude = longitude
        self.full_latitude = latitude
        self.full_x = numpy.arange(len(expected))
        self.full_expected = expected
        self.full_computed = computed
        self.full_difference = computed - expected

        is_floating = issubclass(expected.dtype.type, numpy.floating) or issubclass(
            computed.dtype.type, numpy.floating
        )

        if not is_floating:
            nan_expected = numpy.ma.getmaskarray(expected)
            nan_computed = numpy.ma.getmaskarray(computed)
            self.full_expected = numpy.ma.filled(
                _promote_integer_to_float(expected), fill_value=numpy.nan
            )
            self.full_computed = numpy.ma.filled(
                _promote_integer_to_float(computed), fill_value=numpy.nan
            )
            self.full_difference = numpy.ma.filled(
                _promote_integer_to_float(self.full_difference), fill_value=numpy.nan
            )
        else:
            self.full_expected = numpy.ma.filled(expected, fill_value=numpy.nan)
            self.full_computed = numpy.ma.filled(computed, fill_value=numpy.nan)
            nan_expected = numpy.isnan(self.full_expected)
            nan_computed = numpy.isnan(self.full_computed)
            self.full_difference = numpy.ma.filled(
                self.full_difference, fill_value=numpy.nan
            )

        exclusive_nan = nan_computed ^ nan_expected
        self.exclusive_nan = bool(numpy.any(exclusive_nan))

        # Store NaN distribution data
        self.full_nan_computed: numpy.ndarray | None
        self.full_nan_expected: numpy.ndarray | None
        if self.exclusive_nan:
            self.full_nan_computed = numpy.where(
                exclusive_nan & nan_computed, 0, numpy.nan
            )
            self.full_nan_expected = numpy.where(
                exclusive_nan & nan_expected, 0, numpy.nan
            )
        else:
            self.full_nan_computed = None
            self.full_nan_expected = None

        # Store flags
        self.flags = flag
        self.flag_values: dict[str, numpy.ndarray] = {}
        if flag is not None:
            self.flag_values = {item.varname: item.values for item in flag}

        # Build the data dictionaries
        self._build_sources()

    def _build_sources(
        self,
        indices: numpy.ndarray | None = None,
    ) -> None:
        """Build or rebuild plain data dictionaries.

        Args:
            indices: Optional subset of indices to use (for flag filtering).

        """
        if indices is not None:
            x = self.full_x[indices]
            expected = self.full_expected[indices]
            computed = self.full_computed[indices]
            difference = self.full_difference[indices]
            lon = (
                self.full_longitude[indices]
                if self.full_longitude is not None
                else None
            )
            lat = (
                self.full_latitude[indices] if self.full_latitude is not None else None
            )
        else:
            x = self.full_x
            expected = self.full_expected
            computed = self.full_computed
            difference = self.full_difference
            lon = self.full_longitude
            lat = self.full_latitude

        # Downsample for display if needed
        n = len(x)
        if n > MAX_DISPLAY_POINTS:
            LOGGER.info("Downsampling from %d to %d points", n, MAX_DISPLAY_POINTS)
            ds_x, ds_expected = lttb_downsample(x, expected, MAX_DISPLAY_POINTS)
            _, ds_computed = lttb_downsample(x, computed, MAX_DISPLAY_POINTS)
            _, ds_difference = lttb_downsample(x, difference, MAX_DISPLAY_POINTS)
            display_x = ds_x
            display_expected = ds_expected
            display_computed = ds_computed
            display_difference = ds_difference
            # Use the same downsampled indices for lon/lat
            if lon is not None:
                assert lat is not None
                ds_idx = numpy.searchsorted(x, display_x)
                ds_idx = numpy.clip(ds_idx, 0, len(x) - 1)
                lon = lon[ds_idx]
                lat = lat[ds_idx]
        else:
            display_x = x
            display_expected = expected
            display_computed = computed
            display_difference = difference

        self.source_data: dict[str, Any] = {
            "x": display_x,
            "expected": display_expected,
            "computed": display_computed,
            "difference": display_difference,
        }

        # Coordinates for map display
        if lon is not None:
            self.source_data["longitude"] = lon
            self.source_data["latitude"] = lat

        if self.exclusive_nan:
            if indices is not None:
                assert self.full_nan_computed is not None
                assert self.full_nan_expected is not None
                nan_comp = self.full_nan_computed[indices]
                nan_exp = self.full_nan_expected[indices]
            else:
                nan_comp = self.full_nan_computed
                nan_exp = self.full_nan_expected
            # Downsample NaN data too if needed
            if n > MAX_DISPLAY_POINTS:
                assert nan_comp is not None
                assert nan_exp is not None
                ds_idx = numpy.searchsorted(x, display_x)
                ds_idx = numpy.clip(ds_idx, 0, len(x) - 1)
                nan_comp = nan_comp[ds_idx]
                nan_exp = nan_exp[ds_idx]
            self.source_data["nan_computed"] = nan_comp
            self.source_data["nan_expected"] = nan_exp

        # Build histogram from full (non-downsampled) data
        diff_for_hist = difference[~numpy.isnan(difference)]
        if len(diff_for_hist) > 0:
            histogram, edges = numpy.histogram(diff_for_hist, bins="sturges")
            self.histogram_data: dict[str, Any] = {
                "top": histogram,
                "left": edges[:-1],
                "right": edges[1:],
            }
        else:
            self.histogram_data = {"top": [], "left": [], "right": []}

        # Build statistics from full (non-downsampled) data
        non_nan_diff = difference[~numpy.isnan(difference)]
        abs_non_nan = numpy.abs(non_nan_diff)

        if len(abs_non_nan) > 0:
            self.stats_data: dict[str, Any] = {
                "metric": [
                    "Samples",
                    "Minimum Absolute Error",
                    "Maximum Absolute Error",
                    "Mean Absolute Error",
                    "Mean Squared Error",
                    "Root Mean Squared Error",
                ],
                "value": [
                    len(non_nan_diff),
                    numpy.min(abs_non_nan),
                    numpy.max(abs_non_nan),
                    numpy.mean(abs_non_nan),
                    numpy.mean(abs_non_nan**2),
                    numpy.sqrt(numpy.mean(abs_non_nan**2)),
                ],
            }
        else:
            self.stats_data = {
                "metric": [
                    "Samples",
                    "Minimum Absolute Error",
                    "Maximum Absolute Error",
                    "Mean Absolute Error",
                    "Mean Squared Error",
                    "Root Mean Squared Error",
                ],
                "value": [0, 0, 0, 0, 0, 0],
            }

    def get_zoomed_source_data(self, xmin: float, xmax: float) -> dict[str, Any]:
        """Return source_data re-downsampled within a zoom window.

        When the user zooms tightly, the global LTTB downsampling may leave
        very few points in the visible range. This method re-applies LTTB
        only on the data within [xmin, xmax], ensuring up to
        MAX_DISPLAY_POINTS are always shown in the zoom window.

        Args:
            xmin: Left edge of the zoom window (index value).
            xmax: Right edge of the zoom window (index value).

        Returns:
            A source_data dict with keys x, expected, computed, difference
            (and optionally nan_computed, nan_expected) re-downsampled
            within the zoom window.

        """
        mask = (self.full_x >= xmin) & (self.full_x <= xmax)
        x = self.full_x[mask]
        expected = self.full_expected[mask]
        computed = self.full_computed[mask]
        difference = self.full_difference[mask]

        n = len(x)
        if n > MAX_DISPLAY_POINTS:
            ds_x, ds_expected = lttb_downsample(x, expected, MAX_DISPLAY_POINTS)
            _, ds_computed = lttb_downsample(x, computed, MAX_DISPLAY_POINTS)
            _, ds_difference = lttb_downsample(x, difference, MAX_DISPLAY_POINTS)
            x, expected, computed, difference = (
                ds_x,
                ds_expected,
                ds_computed,
                ds_difference,
            )

        result: dict[str, Any] = {
            "x": x,
            "expected": expected,
            "computed": computed,
            "difference": difference,
        }

        if self.exclusive_nan:
            assert self.full_nan_computed is not None
            assert self.full_nan_expected is not None
            nan_comp = self.full_nan_computed[mask]
            nan_exp = self.full_nan_expected[mask]
            if n > MAX_DISPLAY_POINTS:
                ds_idx = numpy.searchsorted(self.full_x[mask], x)
                ds_idx = numpy.clip(ds_idx, 0, int(numpy.sum(mask)) - 1)
                nan_comp = nan_comp[ds_idx]
                nan_exp = nan_exp[ds_idx]
            result["nan_computed"] = nan_comp
            result["nan_expected"] = nan_exp

        return result

    def apply_flag_filter(self, indices: numpy.ndarray) -> None:
        """Refilter data based on flag selection and rebuild sources.

        Args:
            indices: Array of data point indices to keep.

        """
        self._build_sources(indices)


def _find_coordinates(
    dataset: netCDF4.Dataset,
    varpath: str,
    data_shape: tuple[int, ...],
    active_mask: numpy.ndarray | None,
) -> tuple[numpy.ndarray | None, numpy.ndarray | None]:
    """Find longitude/latitude for a variable.

    Strategy:
      1. Check the ``coordinates`` attribute on the variable itself.
      2. Otherwise, walk up from the variable's group to the root,
         looking for variables whose short name contains "lon"/"lat".

    Returns:
        (longitude, latitude) arrays or (None, None).

    """
    data_size = data_shape[0] if data_shape else 0
    parts = varpath.strip("/").split("/")

    # --- Helper: load a lon/lat pair by absolute paths ----------------
    def _load_pair(
        lon_path: str, lat_path: str
    ) -> tuple[numpy.ndarray, numpy.ndarray] | tuple[None, None]:
        try:
            lon_var = _navigate_to_variable(dataset, lon_path)
            lat_var = _navigate_to_variable(dataset, lat_path)
            lon = numpy.ma.filled(lon_var[:], fill_value=numpy.nan)
            lat = numpy.ma.filled(lat_var[:], fill_value=numpy.nan)
            # Only keep if the size is compatible with the data
            if lon.size != data_size:
                LOGGER.info(
                    "_find_coordinates: size mismatch lon=%d data=%d, skipping",
                    lon.size,
                    data_size,
                )
                return None, None
            if active_mask is not None and lon.shape == active_mask.shape:
                lon = numpy.where(active_mask, lon, numpy.nan)
                lat = numpy.where(active_mask, lat, numpy.nan)
            LOGGER.info("_find_coordinates: loaded lon=%s lat=%s", lon_path, lat_path)
            return lon, lat
        except Exception as exc:
            LOGGER.warning(
                "_find_coordinates: failed to load %s / %s: %s", lon_path, lat_path, exc
            )
            return None, None

    # --- 1. Explicit ``coordinates`` attribute -------------------------
    # The variable's group prefix (e.g. "/data_01" for "/data_01/foo")
    group_prefix = "/" + "/".join(parts[:-1]) if len(parts) > 1 else ""

    try:
        src_var = _navigate_to_variable(dataset, varpath)
        if "coordinates" in src_var.ncattrs():
            coords_attr = str(src_var.getncattr("coordinates"))
            lon_path = lat_path = None
            for coord_path in coords_attr.split():
                # If path starts with "/", it's absolute; otherwise
                # it's relative to the variable's group.
                if not coord_path.startswith("/"):
                    coord_path = f"{group_prefix}/{coord_path}"
                short = coord_path.rsplit("/", 1)[-1].lower()
                if "lon" in short:
                    lon_path = coord_path
                elif "lat" in short:
                    lat_path = coord_path
            if lon_path and lat_path:
                lon, lat = _load_pair(lon_path, lat_path)
                if lon is not None:
                    return lon, lat
    except Exception:
        LOGGER.debug(
            "coordinates attribute lookup failed for %s", varpath, exc_info=True
        )

    # --- 2. Walk up group hierarchy -----------------------------------
    parts = varpath.strip("/").split("/")
    # Start from the variable's own group, then go up to root
    for depth in range(len(parts) - 1, 0, -1):
        group_prefix = "/" + "/".join(parts[:depth])
        try:
            group = dataset
            for p in parts[:depth]:
                group = group.groups[p]
        except KeyError:
            continue

        lon_path = lat_path = None
        for vname in group.variables:
            lower = vname.lower()
            if "longitude" in lower or lower == "lon":
                lon_path = f"{group_prefix}/{vname}"
            elif "latitude" in lower or lower == "lat":
                lat_path = f"{group_prefix}/{vname}"
        if lon_path and lat_path:
            lon, lat = _load_pair(lon_path, lat_path)
            if lon is not None:
                return lon, lat

    LOGGER.info("_find_coordinates: no coordinates found for %s", varpath)
    return None, None


def load_variable_data(
    state: SessionState,
    varpath: str,
) -> DataSource:
    """Load variable data on demand and create a DataSource.

    Args:
        state: The session state holding open datasets.
        varpath: Full path to the variable (e.g., '/data_01/altitude').

    Returns:
        DataSource for the requested variable.

    """
    # Check cache first (move to end for LRU).  Dash dispatches callbacks
    # from multiple threads; guard all cache reads/writes with the lock
    # so concurrent loaders cannot corrupt the OrderedDict's linked list.
    with state.lock:
        if varpath in state.variable_cache:
            LOGGER.info("load_variable_data: CACHE HIT for %s", varpath)
            state.variable_cache.move_to_end(varpath)
            return state.variable_cache[varpath]

    LOGGER.info("load_variable_data: CACHE MISS for %s", varpath)

    # Navigate to the variable in both datasets
    src_var = _navigate_to_variable(state.src_ds, varpath)
    tgt_var = _navigate_to_variable(state.tgt_ds, varpath)

    src_val = src_var[:]
    tgt_val = tgt_var[:]
    LOGGER.info("load_variable_data: raw data shape=%s", src_val.shape)

    # Apply expression mask if active (skip if sizes don't match)
    if state.active_mask is not None:
        if src_val.shape == state.active_mask.shape:
            LOGGER.info(
                "load_variable_data: applying mask (sum=%d/%d)",
                int(state.active_mask.sum()),
                int(state.active_mask.size),
            )
            src_val = _apply_mask(src_val, state.active_mask)
            tgt_val = _apply_mask(tgt_val, state.active_mask)
            LOGGER.info(
                "load_variable_data: after mask, valid points=%d",
                int((~numpy.ma.getmaskarray(src_val)).sum()),
            )
        else:
            LOGGER.info(
                "load_variable_data: mask size %d != data size %d, "
                "skipping mask for %s",
                state.active_mask.size,
                src_val.size,
                varpath,
            )
    else:
        LOGGER.info("load_variable_data: NO active mask")

    # Get relevant flags for this variable's group
    relevant_flags = None
    if state.flags:
        relevant_flags = [f for f in state.flags if f.is_in_current_group(varpath)]
        if not relevant_flags:
            relevant_flags = None

    # Extract lon/lat coordinates
    longitude, latitude = _find_coordinates(
        state.src_ds, varpath, src_val.shape, state.active_mask
    )

    ds = DataSource(
        src_val, tgt_val, relevant_flags, longitude=longitude, latitude=latitude
    )

    # Cache with eviction (another thread may have populated the entry
    # in the meantime; the lock keeps eviction/insertion atomic).
    with state.lock:
        if varpath in state.variable_cache:
            state.variable_cache.move_to_end(varpath)
            return state.variable_cache[varpath]
        if len(state.variable_cache) >= state.max_cache_size:
            oldest_key = next(iter(state.variable_cache))
            del state.variable_cache[oldest_key]
        state.variable_cache[varpath] = ds

    return ds


def list_overlay_candidates(
    state: SessionState,
    varpath: str,
) -> list[str]:
    """List 1-D variable paths in the target dataset, for the overlay dropdown.

    The overlay is rendered on the Signal Comparison plot's secondary y-axis.
    Any 1-D variable is allowed; x values are rescaled to the primary plot's
    index range at render time, so a 1Hz variable can be overlaid on a 20Hz
    plot of the "same" quantity.

    Args:
        state: Session state (uses ``state.tgt_ds``).
        varpath: The current variable path (excluded from results).

    Returns:
        Sorted list of absolute variable paths.

    """
    out: list[str] = []

    def _walk(group: netCDF4.Dataset, prefix: str) -> None:
        for name, var in group.variables.items():
            path = f"{prefix}/{name}"
            if path == varpath:
                continue
            if var.ndim == 1 and var.shape[0] > 1:
                out.append(path)
        for gname, g in group.groups.items():
            _walk(g, f"{prefix}/{gname}")

    _walk(state.tgt_ds, "")
    return sorted(out)


def _find_coordinate_variable(
    dataset: netCDF4.Dataset, varpath: str
) -> netCDF4.Variable | None:
    """Resolve the coordinate variable for a 1-D variable's dimension.

    Looks for a variable whose name equals the variable's first dimension,
    in the same group or any ancestor group up to root. Returns the netCDF4
    Variable or ``None`` if not found.
    """
    try:
        var = _navigate_to_variable(dataset, varpath)
    except (KeyError, IndexError):
        return None
    if var.ndim < 1:
        return None
    dim_name = var.dimensions[0]

    parts = varpath.strip("/").split("/")
    for depth in range(len(parts) - 1, -1, -1):
        group = dataset
        for p in parts[:depth]:
            try:
                group = group.groups[p]
            except KeyError:
                break
        else:
            coord = group.variables.get(dim_name)
            if coord is not None and coord.ndim == 1:
                return coord
    return None


def load_overlay_for_comparison(
    state: SessionState,
    overlay_varpath: str,
    data_source: DataSource,
    primary_varpath: str,
) -> dict[str, Any] | None:
    """Load overlay data for the Signal Comparison plot.

    The overlay is aligned to the primary plot's x-axis (index space) using
    the coordinate variable of each variable's dimension (typically ``time``).
    For each overlay sample, its coordinate value is mapped to a fractional
    index in the primary's coordinate array via linear interpolation. This
    lets a 1Hz variable (e.g. ``mqe_ocean(time)``) overlay correctly onto a
    20Hz plot of the "same" quantity in time.

    Falls back to proportional index rescaling if coordinate variables can't
    be resolved.

    Args:
        state: Session state.
        overlay_varpath: Path of the variable to overlay.
        data_source: DataSource whose x range dictates the overlay's x-span.
        primary_varpath: Path of the primary variable (for coordinate lookup).

    Returns:
        Dict with keys ``x``, ``y``, ``name`` — or ``None`` on load failure.

    """
    try:
        var = _navigate_to_variable(state.tgt_ds, overlay_varpath)
        val = var[:]
    except (KeyError, AttributeError, IndexError) as exc:
        LOGGER.warning(
            "load_overlay_for_comparison: failed to load %s: %s",
            overlay_varpath,
            exc,
        )
        return None

    if val.ndim != 1 or val.shape[0] < 2:
        return None

    if state.active_mask is not None and val.shape == state.active_mask.shape:
        val = _apply_mask(val, state.active_mask)

    if issubclass(val.dtype.type, numpy.integer):
        if isinstance(val, numpy.ma.MaskedArray):
            val = numpy.ma.masked_array(
                _promote_integer_to_float(val.data), mask=val.mask
            )
        else:
            val = _promote_integer_to_float(val)

    if isinstance(val, numpy.ma.MaskedArray):
        arr = numpy.ma.filled(val, fill_value=numpy.nan)
    else:
        arr = numpy.asarray(val, dtype=float)

    m = arr.shape[0]
    primary_n = len(data_source.full_x)

    # Try to align via coordinate variables (e.g. time).
    overlay_coord_var = _find_coordinate_variable(state.tgt_ds, overlay_varpath)
    primary_coord_var = _find_coordinate_variable(state.tgt_ds, primary_varpath)
    x: numpy.ndarray | None = None
    if (
        overlay_coord_var is not None
        and primary_coord_var is not None
        and overlay_coord_var.shape == (m,)
        and primary_coord_var.shape == (primary_n,)
    ):
        t_overlay = numpy.ma.filled(overlay_coord_var[:], fill_value=numpy.nan).astype(
            float
        )
        t_primary = numpy.ma.filled(primary_coord_var[:], fill_value=numpy.nan).astype(
            float
        )
        # Only use coord-based alignment when both arrays are usable and the
        # primary coord is strictly monotonic (required by numpy.interp).
        valid_primary = ~numpy.isnan(t_primary)
        if (
            numpy.all(valid_primary)
            and numpy.all(numpy.diff(t_primary) > 0)
            and not numpy.all(numpy.isnan(t_overlay))
        ):
            x = numpy.interp(
                t_overlay,
                t_primary,
                numpy.arange(primary_n, dtype=float),
                left=numpy.nan,
                right=numpy.nan,
            )
            LOGGER.info(
                "load_overlay_for_comparison: coord-based alignment "
                "(overlay n=%d, primary n=%d)",
                m,
                primary_n,
            )

    if x is None:
        # Fallback: proportional rescaling based on index.
        scale = (primary_n - 1) / (m - 1) if m > 1 else 1.0
        x = numpy.arange(m, dtype=float) * scale

    # Drop samples with NaN x (outside primary's coord range) before LTTB.
    finite = ~numpy.isnan(x)
    if not numpy.all(finite):
        x = x[finite]
        arr = arr[finite]
    if len(x) == 0:
        return None

    if len(x) > MAX_DISPLAY_POINTS:
        x, arr = lttb_downsample(x, arr, MAX_DISPLAY_POINTS)

    return {
        "x": x,
        "y": arr,
        "name": overlay_varpath.rsplit("/", 1)[-1],
    }


def _navigate_to_variable(dataset: netCDF4.Dataset, varpath: str) -> netCDF4.Variable:
    """Navigate to a variable in a NetCDF dataset by its full path.

    Args:
        dataset: The NetCDF dataset.
        varpath: Full path (e.g., '/data_01/altitude').

    Returns:
        The NetCDF variable.

    """
    parts = varpath.strip("/").split("/")
    group = dataset
    for part in parts[:-1]:
        group = group.groups[part]
    return group.variables[parts[-1]]
