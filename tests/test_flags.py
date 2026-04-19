"""Tests for app/flags.py."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy
import pytest

from app.flags import Flag


@pytest.fixture()
def flag_nc(tmp_path: Path) -> Path:
    path = tmp_path / "flag.nc"
    with netCDF4.Dataset(path, "w") as ds:
        grp = ds.createGroup("data_20")
        grp.createDimension("x", 6)
        v = grp.createVariable("surface_class", "i4", ("x",))
        v.flag_values = numpy.array([0, 1, 2], dtype="i4")
        v.flag_meanings = "ocean land ice"
        v[:] = numpy.array([0, 1, 2, 0, 1, 2], dtype="i4")
    return path


def test_flag_construction(flag_nc: Path) -> None:
    with netCDF4.Dataset(flag_nc, "r") as ds:
        flag = Flag("/data_20/surface_class", ds)
    assert flag.varname == "surface_class"
    assert flag.group == "data_20"
    assert flag.options == [
        "surface_class/ocean",
        "surface_class/land",
        "surface_class/ice",
    ]
    assert flag.mapping["surface_class/ocean"] == 0


def test_is_in_current_group(flag_nc: Path) -> None:
    with netCDF4.Dataset(flag_nc, "r") as ds:
        flag = Flag("/data_20/surface_class", ds)
    assert flag.is_in_current_group("/data_20/altitude")
    assert not flag.is_in_current_group("/data_01/altitude")


def test_compute_filtered_indices_single_flag(flag_nc: Path) -> None:
    with netCDF4.Dataset(flag_nc, "r") as ds:
        flag = Flag("/data_20/surface_class", ds)
    idx = flag.compute_filtered_indices(
        ["surface_class/ocean", "surface_class/land"], [flag]
    )
    numpy.testing.assert_array_equal(idx, [0, 1, 3, 4])


def test_mismatched_meanings_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.nc"
    with netCDF4.Dataset(path, "w") as ds:
        grp = ds.createGroup("data_20")
        grp.createDimension("x", 3)
        v = grp.createVariable("flag", "i4", ("x",))
        v.flag_values = numpy.array([0, 1, 2], dtype="i4")
        # one fewer label than values
        v.flag_meanings = "a b"
        v[:] = numpy.array([0, 1, 2], dtype="i4")
    with netCDF4.Dataset(path, "r") as ds, pytest.raises(ValueError):
        Flag("/data_20/flag", ds)
