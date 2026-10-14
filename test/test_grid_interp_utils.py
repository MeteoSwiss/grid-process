# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the grid_interp_utils module"""

import json

from pyproj import CRS
import numpy as np
import xarray as xr

import grid_process.grid_interp_utils as gru
from grid_process.data_model import get_grid_crs


def test_create_dst_grid(data_native_lv95):
    """
    Simple testcase for the create_dst_grid function from grid_interp_utils module
    """

    grid_map_1 = get_grid_crs(data_native_lv95)
    grid_map_2 = json.loads(data_native_lv95.grid_mapping)["epsg_code"]
    bottomleft = (data_native_lv95.y.values[0], data_native_lv95.x.values[0])
    topright = (data_native_lv95.y.values[-1], data_native_lv95.x.values[-1])
    nrowcol = (data_native_lv95.y.size, data_native_lv95.x.size)
    res = np.mean([np.diff(data_native_lv95.y).mean(), np.diff(data_native_lv95.x).mean()])
    dtype = data_native_lv95.x.dtype

    results1 = gru.create_dst_grid(
        grid_map_1,
        bottomleft,
        topright=topright,
        nrowcol=nrowcol,
        add_latlon=True,
        dtype=dtype,
    )
    results2 = gru.create_dst_grid(
        grid_map_2,
        bottomleft,
        topright=topright,
        resolution=res,
        add_latlon=True,
        dtype=dtype,
    )
    results3 = gru.create_dst_grid(grid_map_1, bottomleft, nrowcol=nrowcol, resolution=res)

    expected = data_native_lv95.drop_dims(set(data_native_lv95.dims) - {"x", "y"}).drop_vars("surface_altitude")
    del expected.attrs["source"], expected.attrs["tags"], expected.attrs["grid_mapping"]
    xr.testing.assert_identical(results1, expected.assign_attrs(crs=grid_map_1))
    xr.testing.assert_identical(results2, expected.assign_attrs(crs=grid_map_2))

    expected = expected.drop_vars(("longitude", "latitude"))
    xr.testing.assert_identical(results3, expected.assign_attrs(crs=grid_map_1))

    assert results1.x.dtype == results1.y.dtype == results1.longitude.dtype == results1.latitude.dtype == dtype
    assert results2.x.dtype == results2.y.dtype == results2.longitude.dtype == results2.latitude.dtype == dtype
    assert results3.x.dtype == results3.y.dtype == np.float32


def test_create_dst_grid_from_src(data_native_rotlatlon):
    """
    Simple testcase for the create_dst_grid_from_src function from grid_interp_utils module
    """

    data_native_rotlatlon_reversed = data_native_rotlatlon.isel(y=slice(None, None, -1))
    dst_grid_map = "EPSG:2056"
    results = gru.create_dst_grid_from_src(data_native_rotlatlon, dst_grid_map)
    assert results.sizes["x"] == data_native_rotlatlon.sizes["x"]
    assert results.sizes["y"] == data_native_rotlatlon.sizes["y"]
    assert results.crs == dst_grid_map
    assert results.x.dtype == data_native_rotlatlon.x.dtype

    original = gru.create_dst_grid_from_src(results, get_grid_crs(data_native_rotlatlon))
    assert original.sizes["x"] == data_native_rotlatlon.sizes["x"]
    assert original.sizes["y"] == data_native_rotlatlon.sizes["y"]
    assert original.crs == CRS.from_cf(json.loads(data_native_rotlatlon.grid_mapping))
    assert original.x.dtype == data_native_rotlatlon.x.dtype
    np.testing.assert_allclose(original.x.values, data_native_rotlatlon.x.values, atol=5e-2)
    np.testing.assert_allclose(original.y.values, data_native_rotlatlon.y.values, atol=5e-2)

    # check for src grid with decreasing y coords
    results_reversed = gru.create_dst_grid_from_src(data_native_rotlatlon_reversed, dst_grid_map)
    xr.testing.assert_identical(results_reversed, results.isel(y=slice(None, None, -1)))

    dst_grid_map = "EPSG:4326"
    results = gru.create_dst_grid_from_src(data_native_rotlatlon, dst_grid_map, add_latlon=True)
    assert results.sizes["x"] == data_native_rotlatlon.sizes["x"]
    assert results.sizes["y"] == data_native_rotlatlon.sizes["y"]
    assert results.crs == dst_grid_map
    assert {"longitude", "latitude"}.issubset(results.coords)
    assert results.x.dtype == data_native_rotlatlon.x.dtype
    assert np.allclose(results.x.values, results.longitude.values)
    assert np.allclose(results.y.values[:, np.newaxis], results.latitude.values)

    # check for src grid with decreasing y coords
    results_reversed = gru.create_dst_grid_from_src(data_native_rotlatlon_reversed, dst_grid_map, add_latlon=True)
    xr.testing.assert_identical(results_reversed, results.isel(y=slice(None, None, -1)))
