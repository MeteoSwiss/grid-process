# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the grid_interp_utils module"""

import json

from pyproj import CRS
import numpy as np
import pytest
import xarray as xr

import grid_process.grid_interp_utils as gru
from grid_process.data_model import get_grid_crs
from grid_process.helpers import reproject, sphere_as_wgs84


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
    # rotated pole on a sphere: considered on WGS84 (see grid_process.helpers.sphere_as_wgs84)
    assert original.crs == sphere_as_wgs84(CRS.from_cf(json.loads(data_native_rotlatlon.grid_mapping)))
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


@pytest.mark.parametrize(
    "start, topright, res, expected_n",
    [
        (2600000.0, 2600950.0, 100.0, 10),  # topright not a multiple of the resolution
        (2600000.0, 2601000.0, 100.0, 11),  # topright is a multiple: kept as last point
        (0.0, 0.3, 0.1, 4),  # float truncation edge case (0.3 / 0.1 = 2.9999...)
    ],
)
def test_create_dst_grid_topright_resolution(start, topright, res, expected_n):
    """
    With topright and resolution, the resolution must be exact and topright a closed upper bound
    """

    grid = gru.create_dst_grid(
        "EPSG:2056", (start, start), topright=(topright, topright), resolution=res, dtype=np.dtype("float64")
    )
    for coords in (grid.x.values, grid.y.values):
        assert coords.size == expected_n
        np.testing.assert_allclose(np.diff(coords), res)
        assert coords[0] == start
        assert coords[-1] <= topright + 1e-9


def test_create_dst_grid_missing_args():
    """
    Two out of the three arguments topright, nrowcol, resolution are required
    """

    with pytest.raises(ValueError, match="At least two"):
        gru.create_dst_grid("EPSG:2056", (1.2e6, 2.6e6), resolution=1000.0)


def _lv95_points(grid: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """LV95 coordinates of all latitude/longitude points of a grid"""

    return reproject(grid.longitude.values.ravel(), grid.latitude.values.ravel(), "EPSG:4326", "EPSG:2056")


@pytest.mark.parametrize("src_name", ["data_native_lv95", "data_native_rotlatlon", "model_data", "unstructured"])
def test_create_dst_grid_from_src_bounds(src_name, request):
    """
    The destination grid spans exactly the bounding box of the source domain reprojected in the destination CRS,
    using native metric coordinates when available (identity in the same CRS)
    """

    if src_name == "unstructured":
        src = request.getfixturevalue("data_native_rotlatlon").stack({"cell": ("y", "x")})
    else:
        src = request.getfixturevalue(src_name)
    results = gru.create_dst_grid_from_src(src, "EPSG:2056")

    if src_name == "data_native_lv95":
        np.testing.assert_allclose(results.x.values, src.x.values, rtol=0, atol=1e-6)
        np.testing.assert_allclose(results.y.values, src.y.values, rtol=0, atol=1e-6)
        return

    # all source points are within the destination grid, which is not larger than needed
    x_lv95, y_lv95 = _lv95_points(src)
    atol = 1.0  # float32 coordinates
    np.testing.assert_allclose([results.x.min(), results.x.max()], [x_lv95.min(), x_lv95.max()], rtol=0, atol=atol)
    np.testing.assert_allclose([results.y.min(), results.y.max()], [y_lv95.min(), y_lv95.max()], rtol=0, atol=atol)
    if src_name == "unstructured":
        assert abs(results.sizes["x"] * results.sizes["y"] - src.sizes["cell"]) <= max(results.sizes.values())


@pytest.mark.filterwarnings("ignore:You will likely lose important projection information")  # to_proj4()
@pytest.mark.parametrize("method,kwargs", [("nearest", {}), ("idw", {"k": 4})])
def test_interp2grid_on_created_dst_grid(method, kwargs, data_native_lv95, data_native_rotlatlon):
    """
    interp2grid gives the same results on destination grids created by the utilities, whatever the way their CRS
    is given (string, int, CRS object, WKT, rotated pole on a sphere) and with or without latitude/longitude
    (same values, NaN included)
    """

    import grid_process.grid_interp as gri

    def interp(src: xr.Dataset, dst: xr.Dataset) -> xr.Dataset:
        return gri.interp2grid(src[["test_var_1"]], dst, method, try_reg=False, **kwargs)

    # LV95 destination from a rotated pole source
    grid_kwargs = {"bottomleft": (1.17e6, 2.65e6), "nrowcol": (8, 10), "resolution": 4000.0}
    expected = interp(data_native_rotlatlon, gru.create_dst_grid("EPSG:2056", **grid_kwargs))
    assert expected["test_var_1"].notnull().any()
    lv95 = CRS.from_epsg(2056)
    for crs, add_latlon in [("EPSG:2056", True), (2056, False), (lv95, False), (lv95.to_wkt(), False)]:
        dst = gru.create_dst_grid(crs, add_latlon=add_latlon, **grid_kwargs)
        np.testing.assert_array_equal(
            interp(data_native_rotlatlon, dst)["test_var_1"].values, expected["test_var_1"].values
        )

    # rotated pole destination from an LV95 source
    rotated = get_grid_crs(data_native_rotlatlon)
    dst = gru.create_dst_grid_from_src(data_native_rotlatlon, rotated)
    expected = interp(data_native_lv95, dst)
    assert expected["test_var_1"].notnull().any()
    rotated_sphere = CRS.from_cf(json.loads(data_native_rotlatlon.grid_mapping))
    for other in (
        gru.create_dst_grid_from_src(data_native_rotlatlon, rotated, add_latlon=True),
        dst.assign_attrs(crs=rotated_sphere.to_wkt()),
        dst.assign_attrs(crs=rotated_sphere.to_proj4()),
    ):
        np.testing.assert_array_equal(
            interp(data_native_lv95, other)["test_var_1"].values, expected["test_var_1"].values
        )
