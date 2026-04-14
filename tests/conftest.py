"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy
import pytest


@pytest.fixture()
def tmp_nc_pair(tmp_path: Path):
    """Create a pair of minimal NetCDF files and return their paths.

    The *source* and *target* files share the same structure but
    differ in a single variable so that the comparison logic can be
    exercised.

    Yields:
        ``(src_$1_path, tgt_path)`` as :class:`Path` objects.

    """
    src_path = tmp_path / "ref.nc"
    tgt_path = tmp_path / "tgt.nc"

    # --- Reference ---
    with netCDF4.Dataset(src_path, "w") as ds:
        grp = ds.createGroup("data_01")
        grp.createDimension("x", 10)
        # Variable that will be equal
        v_eq = grp.createVariable("altitude", "f4", ("x",))
        v_eq[:] = numpy.arange(10, dtype="f4")
        # Variable that will differ
        v_diff = grp.createVariable("speed", "f4", ("x",))
        v_diff[:] = numpy.ones(10, dtype="f4")
        # Variable with a scale_factor
        v_tol = grp.createVariable("temperature", "f4", ("x",))
        v_tol.scale_factor = 0.01
        v_tol[:] = numpy.arange(10, dtype="f4") * 0.1
        # Flag variable
        v_flag = grp.createVariable("flag", "i4", ("x",))
        v_flag[:] = numpy.zeros(10, dtype="i4")

    # --- Target ---
    with netCDF4.Dataset(tgt_path, "w") as ds:
        grp = ds.createGroup("data_01")
        grp.createDimension("x", 10)
        v_eq = grp.createVariable("altitude", "f4", ("x",))
        v_eq[:] = numpy.arange(10, dtype="f4")  # same
        v_diff = grp.createVariable("speed", "f4", ("x",))
        v_diff[:] = numpy.ones(10, dtype="f4") * 2.0  # different
        v_tol = grp.createVariable("temperature", "f4", ("x",))
        v_tol.scale_factor = 0.01
        v_tol[:] = numpy.arange(10, dtype="f4") * 0.1 + 0.005  # within tol
        v_flag = grp.createVariable("flag", "i4", ("x",))
        v_flag[:] = numpy.zeros(10, dtype="i4")

    return src_path, tgt_path


@pytest.fixture()
def open_nc_pair(tmp_nc_pair):
    """Open the NetCDF pair and yield ``(src_ds, tgt_ds)``.

    Datasets are closed automatically after the test.
    """
    src_path, tgt_path = tmp_nc_pair
    src_ds = netCDF4.Dataset(src_path, "r")
    tgt_ds = netCDF4.Dataset(tgt_path, "r")
    yield src_ds, tgt_ds
    src_ds.close()
    tgt_ds.close()
