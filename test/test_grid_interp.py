# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the grid_interp module"""

import warnings

import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from scipy.interpolate import griddata


def test_interp_temp():
    """Simple testcase for the interp_temp function from grid_interp module"""

    import grid_process.grid_interp as gri

    # create flat model coords
    x = np.tile(np.arange(0, 100), (100, 1))
    y = x.T
    alt_neighbor = np.arange(25).reshape(5, 5)
    z = np.tile(alt_neighbor, (20, 20))
    z = z + np.floor(np.arange(100) / 5)
    coords_flat = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # create pois coords
    poi_xy = np.arange(0, 100, 5) + 2
    delta_h = np.concatenate((np.linspace(-2, 2, 10), np.arange(-18, -13), np.arange(13, 18)))
    poi_z = 12 + delta_h + np.arange(0, 20)
    poi_coords = np.c_[poi_xy, poi_xy, poi_z]

    # ***case with z_fact=1***

    # create flat model data
    data = 4 + 3 * z
    data[:50, :] += 2 * z[:50, :] ** 2 + z[:50, :] ** 3
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]

    # create expected results
    expected = 4 + 3 * poi_z
    expected[:10] += 2 * poi_z[:10] ** 2 + poi_z[:10] ** 3
    expected = expected[:, None, None]

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=9)

    # compare
    assert np.allclose(expected, results)

    # ***case with pois altitude all >500m above highest grid point***

    # create expected results
    expected = gri.interp_temp_fixgrad(coords_flat, data_flat, poi_coords + 600, r_search=2, z_weight=1)

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords + 600, z_fact=1, k=25)

    # compare
    assert np.allclose(expected, results)

    # ***case with z_fact>>1***

    # create flat model data
    b = 10
    a = 10
    c = 1000
    d = 2000
    data = np.ones(x.shape)
    data[:, -1] = data[:, -1] * np.arange(x.shape[1]) * b + c
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]
    poi_coords_new = poi_coords.copy()
    poi_coords_new[:, -1] = 2981
    z_new = np.zeros(z.shape)
    z_new[:, -1] = np.arange(z.shape[1]) * a + d
    coords_flat_new = np.c_[(x * 10).flatten(), (y * 10).flatten(), z_new.flatten()]

    expected = c - d + b / a * 2981

    # call tested function
    results = gri.interp_temp(coords_flat_new, data_flat, poi_coords_new, z_fact=50, k=25, grad_only=False)

    # compare
    assert np.allclose(expected, results)

    # ***case with grad_only = True and z_fact=1***

    # create flat model data
    data = 1 + 0.008 * z
    data[:50, :] = 8 + 0.008 * z[:50, :]
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]

    # create expected results
    expected = data[poi_xy, poi_xy] + 0.008 * (poi_z - z[poi_xy, poi_xy])
    expected = expected[:, None, None]

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=15, grad_only=True)

    # compare
    assert np.allclose(expected, results)

    # ***case with grad_only = True and z_fact>>1***

    # create flat model data
    b = 0.2
    a = 10
    data = 10.0 + x
    data[-1, :] = data[-1, :] * b
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]
    poi_coords[:, -1] = 506
    z_new = np.zeros(z.shape)
    z_new[-1, :] = np.arange(z.shape[1]) * a + 500
    coords_flat = np.c_[(x * 10).flatten(), (y * 10).flatten(), z_new.flatten()]

    expected = data[-1, :][abs(z_new[-1, :] - 506).argmin()] - b / a * abs(z_new[-1, :] - 506).min()

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=500, k=10, grad_only=True)

    # compare
    assert np.allclose(expected, results)

    # ***case with grad_only = True and grad is clipped to 0.03***

    # create flat model data ()
    b = 5
    a = 10
    data = 10.0 + x
    data[-1, :] = data[-1, :] * b
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]
    poi_coords[:, -1] = 506
    z_new = np.zeros(z.shape)
    z_new[-1, :] = np.arange(z.shape[1]) * a + 500
    coords_flat = np.c_[(x * 10).flatten(), (y * 10).flatten(), z_new.flatten()]

    expected = (
        data[-1, :][abs(z_new[-1, :] - 506).argmin()] - np.clip(b / a, -0.01, 0.03) * abs(z_new[-1, :] - 506).min()
    )

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=500, k=10, grad_only=True)

    # compare
    assert np.allclose(expected, results)

    # ***flat orography: all neighbor altitudes identical (e.g. copies of the pole point of a lat/lon grid)***
    # no sounding can be fitted: fixgrad fallback (nearest grid point corrected with 6 K/km), for both branches

    x_flat = np.tile(np.arange(0, 10), (10, 1))
    y_flat = x_flat.T
    z_flat = np.zeros(x_flat.shape)  # completely flat
    coords_flat_flat = np.c_[x_flat.flatten(), y_flat.flatten(), z_flat.flatten()]
    data_flat_flat = (x_flat + y_flat).flatten().astype(np.float32)[:, None]
    poi_coords_flat = np.array([[2.0, 3.0, 0.0], [7.0, 7.0, 100.0]])  # on grid points: unambiguous nearest

    expected_flat = np.array([[5.0], [14.0 - 0.006 * 100]])
    for grad_only in (False, True):
        results_flat = gri.interp_temp(
            coords_flat_flat, data_flat_flat, poi_coords_flat, z_fact=1, k=9, grad_only=grad_only
        )
        np.testing.assert_allclose(results_flat, expected_flat, rtol=1e-6)

    # ***chunk_size invariance: results must not depend on chunk_size***

    # use the last coords_flat / data_flat / poi_coords from the previous case
    results_default = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=9)
    for cs in (1, 3, len(poi_coords)):
        results_cs = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=9, chunk_size=cs)
        assert np.allclose(results_default, results_cs), f"Results differ with chunk_size={cs}"


def test_sounding_batch():
    """
    Simple testcase for the _sounding_batch function from grid_interp module
    """

    import grid_process.grid_interp as gri

    # create test data — wrap single-POI inputs in a batch dimension of size 1
    x = np.arange(-5, 10, 0.1)

    def func1(x):
        return 2 * np.power(x, 3) + 3 * x - 2

    def func2(x):
        return 2 * x + 4

    y1 = np.tile(func1(x), (4, 6, 10, 1)).T
    y2 = np.tile(func2(x), (4, 6, 10, 1)).T
    y = np.stack((y1, y2), axis=-1)
    x_pred = np.array([3.0])

    # expected results
    y_exp1 = np.ones((10, 6, 4)) * func1(x_pred)
    y_exp2 = np.ones((10, 6, 4)) * func2(x_pred)
    y_exp = np.stack((y_exp1, y_exp2), axis=-1)  # shape (10, 6, 4, 2)

    # _sounding_batch expects (n_sub, k, ...) — wrap in batch dim of 1
    y_batch = y[None]  # (1, k, 10, 6, 4, 2)
    x_batch = x[None]  # (1, k)
    poi_alts = x_pred  # (1,)

    # call tested function
    y_pred = gri._sounding_batch(y_batch, x_batch, poi_alts, deg=3, w=None)

    # compare — output shape is (1, 10, 6, 4, 2)
    assert np.allclose(y_exp, y_pred[0])

    # test rank-deficient fallback: deg=3 with repeated x values falls back to deg=1
    y_rd = np.c_[[3, 3, 3, 3, -1], [2, 2, 2, 2, 2]]  # (5, 2)
    x_rd = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    y_exp_rd = np.array([1.375, 2.0])

    y_rd_batch = y_rd[None]  # (1, 5, 2)
    x_rd_batch = x_rd[None]  # (1, 5)
    poi_alts_rd = np.array([2.5])

    y_pred3 = gri._sounding_batch(y_rd_batch, x_rd_batch, poi_alts_rd, deg=3)
    y_pred1 = gri._sounding_batch(y_rd_batch, x_rd_batch, poi_alts_rd, deg=1)
    np.testing.assert_allclose(y_pred3[0], y_pred1[0], rtol=1e-5)
    np.testing.assert_allclose(y_exp_rd, y_pred1[0], rtol=1e-5)


def test_interp_nearest_alt():
    """
    Simple testcase for the interp_nearest_alt function from grid_interp module
    """

    import grid_process.grid_interp as gri

    # create flat model coords
    x = np.tile(np.arange(0, 1000, 10), (100, 1))
    y = x.T
    z = np.tile(np.arange(0, 100), (100, 1)) * 100
    coords_flat = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # create pois coords
    poi_x = x[0, 40:60:2]
    poi_y = y[40:60:2, 0]
    poi_z = z[0, 40:60:2] + 200
    poi_coords = np.c_[poi_x, poi_y, poi_z]

    # create flat model data
    data = z
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]

    # create expected results
    expected = poi_z
    expected = expected[:, None, None]

    # call tested function
    results = gri.interp_nearest_alt(coords_flat, data_flat, poi_coords, r_search=12, z_weight=500)

    # compare
    assert np.allclose(expected, results)


def test_interp_temp_fixgrad():
    """
    Simple testcase for the interp_temp_fixgrad function from grid_interp
    module
    """

    import grid_process.grid_interp as gri

    grad = 0.006

    # create flat model coords
    x = np.tile(np.arange(10), (10, 1))
    y = x.T
    z = x + 1000
    coords_flat = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # create pois coords
    poi_x = np.array([0, 5, 9])
    poi_z = np.array([900, 1010, 1100])
    poi_coords = np.c_[poi_x, poi_x, poi_z]

    # create flat model data
    data_flat = np.zeros(coords_flat.shape)

    # create expected results
    expected = (z[poi_x, poi_x] - poi_z) * grad
    expected = expected[:, None]

    # call tested function
    results = gri.interp_temp_fixgrad(coords_flat, data_flat, poi_coords, r_search=1, grad=grad)

    # compare
    assert np.allclose(expected, results)


def test_grid_assignation():
    """
    Simple testcase for the _grid_assignation function from grid_interp module
    """

    import grid_process.grid_interp as gri

    # create flat model coords
    x = np.arange(10)
    coords_flat = np.c_[x, x, x]

    # create pois coords
    poi_x = x[4:6]
    poi_coords = np.c_[poi_x, poi_x, poi_x + 2]

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1, z_weight=500, deltah_min=0)

    # compare
    assert np.allclose(poi_x + 1, results)

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1.5, z_weight=500, deltah_min=0)

    # compare
    assert np.allclose(poi_x + 2, results)

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1.5, z_weight=1, deltah_min=0)

    # compare
    assert np.allclose(poi_x, results)

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1.5, z_weight=500, deltah_min=1)

    # compare
    assert np.allclose(poi_x + 1, results)


def test_interp_idw():
    """Simple testcase for the interp_idw function from grid_interp module"""

    import grid_process.grid_interp as gri

    # test interp_idw on test data with method = 'idw_3D'
    weight_power = 2

    # create flat model coords
    x = np.tile(np.arange(0, 100), (100, 1))
    y = x.T
    z = np.ones((100, 100))
    coords_flat = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # create pois coords
    poi_x = np.arange(40, 60, 2) + 0.3
    poi_y = np.arange(40, 60, 2) + 0.8
    poi_z = np.ones(10) + 1
    poi_coords = np.c_[poi_x, poi_y, poi_z]

    # create flat model data
    data = np.random.rand(100, 100)
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]

    # create expected results
    index = np.arange(40, 60, 2)

    top_left_grid = data[index, index]
    top_right_grid = data[index, index + 1]
    bot_left_grid = data[index + 1, index]
    bot_right_grid = data[index + 1, index + 1]

    tmp = np.array([0.8, 0.3]) ** 2
    tmp2 = (1 - np.array([0.8, 0.3])) ** 2
    d = np.sqrt(np.array([tmp[0] + tmp[1], tmp[0] + tmp2[1], tmp2[0] + tmp[1], tmp2[0] + tmp2[1]]) + 1**2)

    weight = np.power(d + np.mean(d) / 50, -weight_power)
    weight = weight / np.sum(weight)

    expected = np.sum(
        np.c_[top_left_grid, top_right_grid, bot_left_grid, bot_right_grid] * weight[np.newaxis, :],
        axis=1,
    )

    expected = expected[:, None, None]

    # call tested function
    results = gri.interp_idw(coords_flat, data_flat, poi_coords, k=4, weight_power=weight_power)

    # compare
    assert np.allclose(expected, results)


def test_interp_upscaled():
    """
    Simple testcase for the interp_upscaled function from grid_interp module
    """

    import grid_process.grid_interp as gri

    # create flat model coords
    x = np.tile(np.arange(0, 100), (100, 1))
    y = x.T
    coords_flat = np.c_[x.flatten(), y.flatten()]

    # create pois coords
    poi_x = np.array([6, 80, 49, 49, 49, 48, 49, 50, 51, 51])
    poi_y = np.array([6, 80, 48, 49, 50, 49, 49, 49, 50, 51])
    poi_coords = np.c_[poi_x, poi_y]

    # create flat model data
    data = np.ones_like(x)
    data[50:, 50:] = 20
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]

    # create expected results
    expected_max = np.array([1, 20, 1, 20, 20, 1, 20, 20, 20, 20])[:, None, None]
    expected_min = np.array([1, 20, 1, 1, 1, 1, 1, 1, 1, 20])[:, None, None]

    # call tested function
    results_max = gri.interp_upscaled(coords_flat, data_flat, poi_coords, k=9, agg_func=np.max)

    results_min = gri.interp_upscaled(coords_flat, data_flat, poi_coords, k=9, agg_func=np.min)

    # compare
    assert np.allclose(expected_max, results_max)
    assert np.allclose(expected_min, results_min)


def test_interp_nearest():
    """
    Simple testcase for the interp_nearest function from grid_interp module
    """

    import grid_process.grid_interp as gri

    # create flat model coords
    coords_flat = np.random.rand(100, 3)

    # create pois coords
    poi_coords = np.tile(np.arange(0, 10), (3, 1)).T

    # create flat model data
    data_flat = np.random.rand(100)

    # create expected results
    expected = griddata(coords_flat, data_flat, poi_coords, "nearest")

    # call tested function
    results = gri.interp_nearest(coords_flat, data_flat, poi_coords)

    # compare
    assert np.allclose(expected, results)


def test_interp_snowlim():
    """
    Simple testcase for the interp_snowlim function from grid_interp module
    """

    import grid_process.grid_interp as gri

    # create flat model coords
    x = np.tile(np.arange(0, 10), (10, 1))
    y = x.T
    z = y + 1000
    coords_flat = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # create pois coords
    poi_x = np.array([2, 2, 2, 8, 8, 8])
    poi_y = np.array([0, 3, 7, 0, 3, 7])
    poi_z = np.array([150, 300, 320, 390, 1000, 2000])
    poi_coords = np.c_[poi_x, poi_y, poi_z]

    # create flat model data
    data = np.ones_like(x) * np.nan
    data = np.broadcast_to(data[:, :, None, None], x.shape + (6, 5)).copy()
    data[:5, 5, :, :2] = 950
    data[5:, 5, :, :2] = 1200
    data[4:6, 4:6, :, 3:] = np.array([[1002, 1004], [1004, 1006]])[:, :, None, None]

    data_flat = data.reshape((-1, 6, 5)).astype(np.float32)

    # create expected results
    exp_1 = np.array([950, 950, 1006, 950, 950, 1006])
    exp_2 = np.array([1000, 1002, 1004, 1000, 1002, 1006])
    min_vals = np.array([0, 100, 100, 150, 200, 200])[:, None, None]
    expected = (np.zeros((6, 6, 5))).astype(np.float32) + min_vals
    expected[:, :, :2] = exp_1[:, None, None]
    expected[:, :, 3:] = exp_2[:, None, None]

    # call tested function
    results = gri.interp_snowlim(coords_flat, data_flat, poi_coords, k=5, d_max=40000)
    # compare
    assert np.allclose(expected, results)
    assert results.dtype == data_flat.dtype

    # test with k larger than number of available gridpoints
    # call tested function
    results = gri.interp_snowlim(coords_flat, data_flat, poi_coords, k=coords_flat.shape[0] + 1, d_max=40000)
    # compare
    assert np.allclose(expected.clip(max=min(coords_flat[:, -1])), results)
    assert results.dtype == data_flat.dtype

    # test with small d_max
    # create expected results
    exp_2 = np.array([0, 1002, 1004, 150, 1002, 1006])
    expected[:, :, 3:] = exp_2[:, None, None]

    # call tested function
    results = gri.interp_snowlim(coords_flat, data_flat, poi_coords, k=5, d_max=4)
    # compare
    assert np.allclose(expected, results)
    assert results.dtype == data_flat.dtype

    # call tested function
    results = gri.interp_snowlim(coords_flat, data_flat, poi_coords, k=5, d_max=1)
    # compare
    assert np.allclose(min_vals, results)

    # check that algo also works when no NaNs are present
    data_flat = np.zeros_like(data_flat) + 5000
    results = gri.interp_snowlim(coords_flat, data_flat, poi_coords, k=5)
    # compare
    assert np.allclose(5000, results)


def test_fillna_alt():
    """Simple testcase for the _fillna_alt function from grid_interp module"""

    import grid_process.grid_interp as gri

    # create flat model coords
    x = np.arange(10)
    coords_flat = np.c_[x, x, x + 1000]

    # create pois coords
    poi_x = np.array([2, 3, 4])
    poi_y = np.array([7, 3, 2])
    poi_coords = np.c_[poi_x, poi_y, poi_y]

    # create flat model data
    data_flat = np.ones(coords_flat.shape) * np.nan

    # call tested function
    results = gri._fillna_alt(coords_flat, data_flat, poi_coords, k=len(x))

    # compare
    assert np.allclose(coords_flat[:, -1].min(), results)


def test_compute_dist_weight():
    """
    Simple testcase for the _compute_dist_weight function from grid_interp
    module
    """

    import grid_process.grid_interp as gri

    dist = np.random.rand(10, 5)
    weights = gri._compute_dist_weight(dist, 2, np.float32)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    weights2 = gri._compute_dist_weight(dist, 0, np.float64)
    assert np.all(weights2 == 1 / 5)


def test_nn_lookup():
    """
    Simple testcase for the _compute_dist_weight function from grid_interp
    module
    """

    import grid_process.grid_interp as gri

    k = 4

    # create flat model coords
    x = np.tile(np.arange(0, 10), (10, 1))
    y = x.T
    coords_flat = np.c_[x.flatten(), y.flatten()]

    # create pois coords
    poi_x = np.array([0.1, 5.6])
    poi_y = np.array([0.3, 8.7])
    poi_coords = np.c_[poi_x, poi_y]

    # create expected results
    exp_ii_1 = np.ravel_multi_index((np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])), x.shape)
    exp_ii_2 = np.ravel_multi_index((np.array([9, 9, 8, 8]), np.array([6, 5, 6, 5])), x.shape)
    expected_ii = np.vstack((exp_ii_1, exp_ii_2))
    expected_dd = np.array(
        [
            [np.linalg.norm(point - coords_flat[ind]) for ind in expected_ii[ind_poi]]
            for ind_poi, point in enumerate(poi_coords)
        ]
    )

    # call tested function
    results_dd, results_ii = gri.nn_lookup(coords_flat, poi_coords, k)

    # compare
    assert np.allclose(expected_dd, results_dd)
    assert np.allclose(expected_ii, results_ii)


def test_flatten_concat(model_data, poi_def):
    """
    Simple testcase for the _flatten_concat function from grid_interp module
    """

    import grid_process.grid_interp as gri
    from grid_process.ds_utils import ds_to_da
    from grid_process.data_model import get_spatial_dims
    from grid_process import data_model as dm

    methods_1 = ("nearest", "linear", "cubic", "idw", "upscaled")
    methods_2 = (
        "temp",
        "temp_fixgrad",
        "nearest_alt",
        "snowlim",
        "idw_3D",
        "nearest_3D",
        "linear_3D",
        "upscaled_3D",
    )
    spatial_dims = get_spatial_dims(model_data)

    ind_nan_x = xr.DataArray([1, 3], dims=["z"])
    ind_nan_y = xr.DataArray([2, 4], dims=["z"])

    da, vars_attrs = ds_to_da(model_data)
    da = da.transpose("y", "x", ...)

    da[dict(y=ind_nan_y, x=ind_nan_x)] = np.nan
    da[dict(x=5, y=5, lead_time=5)] = np.nan

    n_nan = len(ind_nan_x)
    n_grids = model_data.sizes["y"] * model_data.sizes["x"] - n_nan
    n_pois = len(poi_def)
    trail_shape = da.shape[2:]
    ind_flat_nan = np.ravel_multi_index((5, 5), da.shape[:2]) - n_nan

    for method in methods_1 + methods_2:
        model_coords_flat, model_data_flat, pois_coords = gri._flatten_concat(da, spatial_dims, poi_def, method)

        n_coords = 2 if method in methods_1 else 3
        all_nan_bool = np.all(np.isnan(model_data_flat), axis=tuple(range(1, len(trail_shape) + 1)))

        if method == "snowlim":
            assert model_coords_flat.shape == (n_grids + n_nan, n_coords)
            assert np.all(np.isnan(model_data_flat[ind_flat_nan + n_nan, :, :, 5]))
            assert model_data_flat.shape == (n_grids + n_nan,) + trail_shape
            assert all_nan_bool.any()
        else:
            assert model_coords_flat.shape == (n_grids, n_coords)
            assert np.all(np.isnan(model_data_flat[ind_flat_nan, :, :, 5]))
            assert model_data_flat.shape == (n_grids,) + trail_shape
            assert not all_nan_bool.any()

        assert pois_coords.shape == (n_pois, n_coords)

        # grid and POIs both in lat/lon over Switzerland: coords transformed to 'EPSG:2056' (CH1903+/LV95) for all
        # methods, so that the nearest neighbor searches and triangulations use distances in meters
        assert (pois_coords[:, :2] > 1e6).all() and (pois_coords[:, :2] < 3e6).all()
        assert (model_coords_flat[:, :2] > 1e6).all() and (model_coords_flat[:, :2] < 3e6).all()
        if method in methods_2:
            # altitude as last column
            np.testing.assert_array_equal(pois_coords[:, -1], dm.get_poi_alt(poi_def))


@pytest.mark.parametrize(
    "model_data_2dflat",
    ["2d_grid", "flat_grid"],
    indirect=True,
)
def test_interp2poi(model_data_2dflat, poi_def):
    """Simple testcase for the interp2poi function from grid_interp module"""

    import grid_process.grid_interp as gri

    # unregular grid interpolation
    results_nn = gri.interp2poi(model_data_2dflat, poi_def, "nearest")
    if "cell" in model_data_2dflat.dims:  # as GRIB read with cfgrib/earthkit-data (no coordinate)
        results_values = gri.interp2poi(
            model_data_2dflat.drop_vars(("cell", "y", "x")).rename_dims(cell="values"), poi_def, "nearest"
        )
        xr.testing.assert_identical(results_values, results_nn.drop_vars(("y", "x"), errors="ignore"))

    # check for conserved dimensions
    model_data_dims = {key: value for key, value in model_data_2dflat.sizes.items() if key not in ("x", "y", "cell")}
    results_nn_dims = {key: value for key, value in results_nn.sizes.items() if key != "poi"}
    assert model_data_dims == results_nn_dims
    assert results_nn.attrs == {"source": model_data_2dflat.source}

    # check for conserved variables and variables attributes
    model_vars_attrs = {varname: da.attrs for varname, da in model_data_2dflat.items()}
    res_vars_attrs = {varname: da.attrs for varname, da in results_nn.items()}
    assert res_vars_attrs == model_vars_attrs

    # nearest regular grid interpolation
    if "cell" in model_data_2dflat.dims:
        model_data_2dflat = model_data_2dflat.drop_vars(("cell", "y", "x"))
        coords = {"y": model_data_2dflat.latitude, "x": model_data_2dflat.longitude}
    else:
        coords = {
            "y": model_data_2dflat.latitude[:, 0].values,
            "x": model_data_2dflat.longitude[0, :].values,
        }

    regulargrid_data = model_data_2dflat.assign_coords(coords).assign_attrs(crs="EPSG:4326")

    results_nn_reg = regulargrid_data.grid_interp.interp2poi(poi_def, "nearest")

    # check for conserved dimensions
    regulargrid_data_dims = {
        key: value for key, value in regulargrid_data.sizes.items() if key not in ("x", "y", "cell")
    }
    results_nn_reg_dims = {key: value for key, value in results_nn_reg.sizes.items() if key != "poi"}
    assert regulargrid_data_dims == results_nn_reg_dims
    assert results_nn_reg.source == regulargrid_data.source

    # check for conserved variables and variables attributes
    regulargrid_vars_attrs = {varname: da.attrs for varname, da in regulargrid_data.items()}
    res_reg_vars_attrs = {varname: da.attrs for varname, da in results_nn_reg.items()}
    assert res_reg_vars_attrs == regulargrid_vars_attrs

    # check that both regular and irregular interpolation returned the same
    assert np.allclose(results_nn.to_array(), results_nn_reg.to_array())

    # check for the number of pois
    assert len(results_nn.poi) == len(results_nn_reg.poi) == len(poi_def)

    # 3D methods
    results_idw_3d = gri.interp2poi(model_data_2dflat, poi_def, "idw_3D", k=16, weight_power=0.5)
    results_upscaled_3d = gri.interp2poi(model_data_2dflat, poi_def, "upscaled_3D", k=16, agg_func=np.max)

    assert (results_upscaled_3d.to_array() >= results_idw_3d.to_array()).all()

    # input dataset with only NaNs should return a dataset with NaNs
    model_data_nan = xr.full_like(model_data_2dflat, np.nan)
    results_nan = gri.interp2poi(model_data_nan, poi_def, "nearest")

    # check for correct 'poi' dimension and only NaNs in the output
    assert "poi" in results_nan.dims
    assert np.all(results_nan.to_array().isnull())

    # check for conserved dimensions
    results_nan_dims = {key: value for key, value in results_nan.sizes.items() if key != "poi"}
    assert model_data_dims == results_nan_dims
    assert results_nan.attrs == {"source": model_data_2dflat.source}

    # check for conserved variables and variables attributes
    res_vars_attrs = {varname: da.attrs for varname, da in results_nan.items()}
    assert res_vars_attrs == model_vars_attrs

    # check consistency with other interpolation methods
    assert results_nan.equals(gri.interp2poi(model_data_nan, poi_def, "linear"))

    # input dataset with only NaNs should return values < 200m for "snowlim"
    results_nan = gri.interp2poi(model_data_nan, poi_def, "snowlim", k=9)

    # check for correct 'poi' dimension and correct values in the output
    assert "poi" in results_nan.dims
    assert np.all(results_nan.to_array() <= 200)

    # check for conserved dimensions
    results_nan_dims = {key: value for key, value in results_nan.sizes.items() if key != "poi"}
    assert model_data_dims == results_nan_dims
    assert results_nan.attrs == {"source": model_data_2dflat.source}

    # check for conserved variables and variables attributes
    res_vars_attrs = {varname: da.attrs for varname, da in results_nan.items()}
    assert res_vars_attrs == model_vars_attrs

    # input dataset with few non-NaNs should return values for all POIs
    model_data_fewnan = model_data_nan.copy()
    if "cell" in model_data_fewnan.dims:
        model_data_fewnan[dict(cell=slice(24, 50))] = 1
    else:
        model_data_fewnan[dict(x=slice(1, 4), y=slice(1, 4))] = 1

    results_fewnan = gri.interp2poi(model_data_fewnan, poi_def, "idw", k=4)

    # check for correct values in the output
    assert np.allclose(results_fewnan.to_array(), 1)

    # test with a single POI
    results_nn = gri.interp2poi(model_data_2dflat, poi_def.loc[[0]], "nearest")
    assert results_nn.sizes["poi"] == 1

    # check for conserved dimensions
    results_nn_dims = {key: value for key, value in results_nn.sizes.items() if key != "poi"}
    assert model_data_dims == results_nn_dims
    assert results_nn.attrs == {"source": model_data_2dflat.source}

    # check for conserved variables and variables attributes
    res_vars_attrs = {varname: da.attrs for varname, da in results_nn.items()}
    assert res_vars_attrs == model_vars_attrs


@pytest.mark.parametrize(
    "data_native_rotlatlon_2dflat,data_native_lv95_2dflat",
    [("2d_grid", "2d_grid"), ("flat_grid", "flat_grid")],
    indirect=True,
)
def test_interp2poi_native_coords(
    data_native_rotlatlon_2dflat,
    data_native_lv95_2dflat,
    poi_def_rotlatlon,
    poi_def_lv95,
):
    """Simple testcase for the interp2poi function from grid_interp module with native coordinates"""

    import grid_process.grid_interp as gri

    # unregular grid interpolation with y/x
    results_nn_native = gri.interp2poi(
        data_native_rotlatlon_2dflat.drop_vars(["longitude", "latitude"]),
        poi_def_rotlatlon,
        "nearest",
        try_reg=False,
    )
    # regular grid interpolation
    results_nn_reg = gri.interp2poi(
        data_native_rotlatlon_2dflat.drop_vars(["longitude", "latitude"]),
        poi_def_rotlatlon,
        "nearest",
        try_reg=True,
    )
    # check all are equivalent
    xr.testing.assert_allclose(results_nn_native, results_nn_reg)

    # unregular grid interpolation with y/x
    results_linear_native = gri.interp2poi(
        data_native_lv95_2dflat.drop_vars(["longitude", "latitude"]),
        poi_def_lv95,
        "linear",
        try_reg=False,
    )
    # regular grid interpolation
    results_linear_reg = gri.interp2poi(
        data_native_lv95_2dflat.drop_vars(["longitude", "latitude"]),
        poi_def_lv95,
        "linear",
        try_reg=True,
    )
    # check both are equivalent
    xr.testing.assert_allclose(results_linear_native, results_linear_reg)

    # check for conserved dimensions
    data_dims = {key: value for key, value in data_native_lv95_2dflat.sizes.items() if key not in ("x", "y", "cell")}
    results_dims = {key: value for key, value in results_linear_native.sizes.items() if key != "poi"}
    assert data_dims == results_dims
    assert results_linear_native.attrs == {"source": data_native_lv95_2dflat.source}

    # check for conserved variables and variables attributes
    model_vars_attrs = {varname: da.attrs for varname, da in data_native_lv95_2dflat.items()}
    res_vars_attrs = {varname: da.attrs for varname, da in results_linear_native.items()}
    assert res_vars_attrs == model_vars_attrs

    # check for conserved variables and variables attributes
    model_vars_attrs = {varname: da.attrs for varname, da in data_native_lv95_2dflat.items()}
    res_vars_attrs = {varname: da.attrs for varname, da in results_linear_native.items()}
    assert res_vars_attrs == model_vars_attrs

    # check for the number of pois
    assert len(results_linear_native.poi) == len(results_nn_reg.poi) == len(poi_def_lv95)

    # methods with height
    # using lon/lat
    results_nn_3d = gri.interp2poi(data_native_lv95_2dflat, poi_def_lv95, "nearest_3D")
    # using x/y
    results_nn_3d_native = gri.interp2poi(
        data_native_lv95_2dflat.drop_vars(["longitude", "latitude"]),
        poi_def_lv95,
        "nearest_3D",
    )
    xr.testing.assert_allclose(results_nn_3d, results_nn_3d_native)

    results_temp_native = gri.interp2poi(
        data_native_rotlatlon_2dflat.drop_vars(["longitude", "latitude"]),
        poi_def_rotlatlon,
        "temp",
        k=36,
        weight_power=0.5,
    )

    # check for conserved dimensions
    data_dims = {
        key: value for key, value in data_native_rotlatlon_2dflat.sizes.items() if key not in ("x", "y", "cell")
    }
    results_temp_dims = {key: value for key, value in results_temp_native.sizes.items() if key != "poi"}
    assert data_dims == results_temp_dims
    assert results_temp_native.attrs == {"source": data_native_rotlatlon_2dflat.source}

    # check for conserved variables and variables attributes
    model_vars_attrs = {varname: da.attrs for varname, da in data_native_rotlatlon_2dflat.items()}
    res_vars_attrs = {varname: da.attrs for varname, da in results_temp_native.items()}
    assert res_vars_attrs == model_vars_attrs

    # check for conserved variables and variables attributes
    model_vars_attrs = {varname: da.attrs for varname, da in data_native_rotlatlon_2dflat.items()}
    res_vars_attrs = {varname: da.attrs for varname, da in results_temp_native.items()}
    assert res_vars_attrs == model_vars_attrs


@pytest.mark.parametrize(
    "model_data_2dflat,data_native_lv95_2dflat",
    [("2d_grid", "2d_grid"), ("flat_grid", "flat_grid")],
    indirect=True,
)
def test_slice_boundingbox(model_data_2dflat, poi_def, data_native_lv95_2dflat, poi_def_lv95):
    """Simple testcase for the slice_boundingbox function from grid_interp module"""

    import grid_process.grid_interp as gri

    # check that with a single POI the sliced grid has correct sizes. Both grids are sliced in latitude/longitude
    # (~46°N), where the x buffer is widened by 1 / cos(latitude): 1 -> 2 columns on each side
    for data, pois in ((model_data_2dflat, poi_def), (data_native_lv95_2dflat, poi_def_lv95)):
        sliced_data = gri.slice_boundingbox(data, pois.loc[[0]], method="nearest")
        if "cell" in sliced_data.dims:
            assert 4 <= sliced_data.sizes["cell"] <= 24
        else:
            assert sliced_data.sizes["x"] == 4
            assert sliced_data.sizes["y"] == 2


@pytest.mark.parametrize("method", ["nearest", "linear"])
def test_interp_regular(method, model_data, poi_def):
    """
    Simple testcase for the _interp_regular function from grid_interp module
    """

    import grid_process.grid_interp as gri

    regulargrid_data = (
        model_data.assign_coords(
            {
                "y": model_data.latitude[:, 0].values,
                "x": model_data.longitude[0, :].values,
            }
        )
        .assign_attrs(crs="EPSG:4326")
        .astype(np.int16)
    )

    results = gri._interp_regular(regulargrid_data, ("y", "x"), poi_def, method)
    assert all([da.dtype == np.int16 for _, da in results.items()])

    if method == "nearest":
        expected = regulargrid_data.sel(
            x=xr.DataArray(poi_def.geometry.x.values, dims="poi"),
            y=xr.DataArray(poi_def.geometry.y.values, dims="poi"),
            method="nearest",
        )
        assert np.allclose(results.to_array(), expected.to_array())


@pytest.mark.parametrize(
    "model_data_2dflat",
    ["2d_grid", "flat_grid"],
    indirect=True,
)
def test_interp2grid(model_data_2dflat, data_native_lv95, data_native_rotlatlon):
    """Simple testcase for the interp2grid function from grid_interp module"""

    import grid_process.grid_interp as gri

    if "cell" in model_data_2dflat.dims:
        model_data_2dflat = model_data_2dflat.drop_vars(("cell", "y", "x"))
        coords = {"y": model_data_2dflat.latitude, "x": model_data_2dflat.longitude}
    else:
        coords = {
            "y": model_data_2dflat.latitude[:, 0].values,
            "x": model_data_2dflat.longitude[0, :].values,
        }

    model_data_2dflat = model_data_2dflat.assign_coords(coords).assign_attrs(grid_mapping={"epsg_code": "EPSG:4326"})

    def assert_same_as_regular(regular: xr.Dataset, generic: xr.Dataset, method: str) -> None:
        """
        The generic path triangulates in LV95 (meters), the regular path interpolates bilinearly in lat/lon: since
        the test data are linear in lat/lon, 'linear' differs by less than 0.2 % of the grid step along x (528 for
        test_var_1, 1320 for test_var_2).
        """

        if method == "linear":
            xr.testing.assert_allclose(regular, generic, rtol=0, atol=2.5)
            assert regular.attrs == generic.attrs
        else:
            xr.testing.assert_identical(regular, generic)

    for method in ["nearest", "linear", "nearest_3D"]:
        # unregular grid interpolation with lat/lon
        results = gri.interp2grid(model_data_2dflat, data_native_lv95, method, try_reg=False)
        assert results.grid_mapping == data_native_lv95.grid_mapping
        assert results.source == model_data_2dflat.source
        assert results.test_var_1.dtype == model_data_2dflat.test_var_1.dtype
        assert (results.x.values == data_native_lv95.x.values).all()
        assert (results.y.values == data_native_lv95.y.values).all()
        # check for conserved dimensions
        data_dims = {key: value for key, value in data_native_lv95.sizes.items()}
        results_dims = {key: value for key, value in results.sizes.items()}
        assert data_dims == results_dims

        # check for conserved variables and variables attributes
        model_vars_attrs = {varname: da.attrs for varname, da in model_data_2dflat.items()}
        res_vars_attrs = {varname: da.attrs for varname, da in results.items()}
        assert res_vars_attrs == model_vars_attrs

        # check preserved dtype
        results_int16 = gri.interp2grid(model_data_2dflat.astype(np.int16), data_native_lv95, method, try_reg=False)
        assert results_int16.test_var_1.dtype == np.int16

        # regular grid interpolation with y/x and lat/lon for dst_grid
        results_regular = gri.interp2grid(model_data_2dflat, data_native_lv95, method, try_reg=True)
        assert_same_as_regular(results_regular, results, method)

        # regular grid interpolation with y/x and y/x for dst_grid
        results_regular_2 = gri.interp2grid(
            model_data_2dflat,
            data_native_lv95.drop_vars(["longitude", "latitude"]),
            method,
            try_reg=True,
        )

        if method == "linear" and "cell" in model_data_2dflat.dims:
            results_regular_2.update(results.drop_vars(["longitude", "latitude"]))

        assert_same_as_regular(results_regular_2, results.drop_vars(["longitude", "latitude"]), method)

    # same with dst_grid in rotated lat lon native grid.
    for method in ["nearest", "linear", "nearest_3D"]:
        # unregular grid interpolation with y/x and lat/lon for dst_grid
        results = model_data_2dflat.drop_vars(["latitude", "longitude"]).grid_interp.interp2grid(
            data_native_rotlatlon, method, try_reg=False
        )
        assert results.grid_mapping == data_native_rotlatlon.grid_mapping
        assert results.source == model_data_2dflat.source
        assert (results.x.values == data_native_rotlatlon.x.values).all()
        assert (results.y.values == data_native_rotlatlon.y.values).all()
        # check for conserved dimensions
        data_dims = {key: value for key, value in data_native_rotlatlon.sizes.items()}
        results_dims = {key: value for key, value in results.sizes.items()}
        assert data_dims == results_dims

        # check for conserved variables and variables attributes
        model_vars_attrs = {varname: da.attrs for varname, da in model_data_2dflat.items()}
        res_vars_attrs = {varname: da.attrs for varname, da in results.items()}
        assert res_vars_attrs == model_vars_attrs

        # regular grid interpolation with y/x and lat/lon for dst_grid
        results_regular = gri.interp2grid(model_data_2dflat, data_native_rotlatlon, method, try_reg=True)
        # 'linear': triangulation in LV95 vs bilinear in lat/lon, see assert_same_as_regular
        atol = 2.5 if method == "linear" else 1e-8
        xr.testing.assert_allclose(results.where(results_regular.notnull()), results_regular, rtol=1e-5, atol=atol)

    data_flat = data_native_lv95.stack(poi=("y", "x"))
    poi_def = data_flat.poi.to_dataframe()[["latitude", "longitude"]].reset_index()
    poi_def = gpd.GeoDataFrame(
        poi_def,
        geometry=gpd.points_from_xy(poi_def.longitude, poi_def.latitude),
    ).drop(columns=["latitude", "longitude", "x", "y"])
    poi_def.crs = "epsg:4326"
    poi_def["name"] = poi_def.index.values.astype(str)
    poi_def["height_masl"] = data_flat.surface_altitude.to_dataframe()["surface_altitude"].values[:, 0]

    results_temp = gri.interp2grid(
        model_data_2dflat,
        data_native_lv95,
        "temp_fixgrad",
        r_search=2,
    )
    assert results_temp.grid_mapping == data_native_lv95.grid_mapping
    assert results_temp.source == model_data_2dflat.source
    assert results_temp.test_var_1.dtype == model_data_2dflat.test_var_1.dtype
    assert (results_temp.x.values == data_native_lv95.x.values).all()
    assert (results_temp.y.values == data_native_lv95.y.values).all()
    # check for conserved dimensions
    data_dims = {key: value for key, value in data_native_lv95.sizes.items()}
    results_dims = {key: value for key, value in results_temp.sizes.items()}
    assert data_dims == results_dims

    # check for conserved variables and variables attributes
    model_vars_attrs = {varname: da.attrs for varname, da in model_data_2dflat.items()}
    res_vars_attrs = {varname: da.attrs for varname, da in results_temp.items()}
    assert res_vars_attrs == model_vars_attrs

    results_temp_poi = gri.interp2poi(
        model_data_2dflat,
        poi_def,
        "temp_fixgrad",
        r_search=2,
    )

    np.testing.assert_allclose(
        results_temp.stack(poi=("y", "x")).transpose("poi", ...).to_array().values,
        results_temp_poi.to_array().values,
        rtol=1e-06,
    )


@pytest.mark.parametrize(
    "data_cf_2dflat",
    ["2d_grid", "flat_grid", "y_decreasing"],
    indirect=True,
)
def test_extended_datamodel(data_cf_2dflat, poi_def_rotlatlon):
    """
    Simple testcase for the interp2poi and interp2grid functions from grid_interp
    module with the extended data model, including when input dataset follow CF
    conventions to encode the grid_mapping
    """

    import grid_process.grid_interp as gri

    # unregular poi interpolation
    results_nn = gri.interp2poi(data_cf_2dflat, poi_def_rotlatlon, "idw", k=4, try_reg=False)

    # check for conserved dimensions
    model_data_dims = {key: value for key, value in data_cf_2dflat.sizes.items() if key not in ("x_1", "y_1", "points")}
    results_nn_dims = {key: value for key, value in results_nn.sizes.items() if key != "poi"}
    assert model_data_dims == results_nn_dims

    # check for conserved variables and variables attributes
    model_vars_attrs = {
        varname: {key: val for key, val in da.attrs.items() if "grid_mapping" not in key}
        for varname, da in data_cf_2dflat.items()
        if "grid_mapping" not in varname
    }
    res_vars_attrs = {varname: da.attrs for varname, da in results_nn.items()}
    assert res_vars_attrs == model_vars_attrs

    # check for global attrs
    assert "institution" in results_nn.attrs
    assert "history" in results_nn.attrs

    # regular poi interpolation
    results_nn_reg = gri.interp2poi(data_cf_2dflat, poi_def_rotlatlon, "linear", try_reg=True)

    # check for conserved dimensions
    results_nn_reg_dims = {key: value for key, value in results_nn_reg.sizes.items() if key != "poi"}
    assert model_data_dims == results_nn_reg_dims

    # check for conserved variables and variables attributes

    res_reg_vars_attrs = {varname: da.attrs for varname, da in results_nn_reg.items()}
    assert res_reg_vars_attrs == res_vars_attrs

    # check for global attrs
    assert "institution" in results_nn_reg.attrs
    assert "history" in results_nn_reg.attrs

    # grid to grid interpolation
    if "points" in data_cf_2dflat.dims:
        dst_grid = data_cf_2dflat.unstack()
        lonlat = ("lon", "lat")
    else:
        dst_grid = data_cf_2dflat.drop_vars("grid_mapping_dummy")
        lonlat = ("lon_1", "lat_1")

    dst_grid = (
        dst_grid.assign_coords(
            {
                "x_1": dst_grid[lonlat[0]][0, :].values + 0.05,
                "y_1": dst_grid[lonlat[1]][:, 0].values - +0.06,
            }
        )
        .drop_vars(lonlat)
        .rename(x_1="longitude", y_1="latitude")
    )
    for varname in dst_grid:
        del dst_grid[varname].attrs["grid_mapping"]

    results = gri.interp2grid(data_cf_2dflat, dst_grid, method="temp_fixgrad", r_search=1.5)
    assert results.test_var_1.dtype == data_cf_2dflat.test_var_1.dtype
    assert (results.longitude.values == dst_grid.longitude.values).all()
    assert (results.latitude.values == dst_grid.latitude.values).all()
    # check for conserved dimensions
    data_dims = {key: value for key, value in dst_grid.sizes.items()}
    results_dims = {key: value for key, value in results.sizes.items()}
    assert data_dims == results_dims

    # check for conserved variables and variables attributes
    model_vars_attrs = {
        varname: {key: val for key, val in da.attrs.items() if "grid_mapping" not in key}
        for varname, da in data_cf_2dflat.items()
        if "grid_mapping" not in varname
    }
    res_vars_attrs = {varname: da.attrs for varname, da in results.items()}
    assert res_vars_attrs == model_vars_attrs

    # check for global attrs
    assert "institution" in results.attrs
    assert "history" in results.attrs

    # check preserved dtype
    results_float64 = gri.interp2grid(data_cf_2dflat.astype(np.float64), dst_grid, "linear", try_reg=True)
    assert results_float64.test_var_1.dtype == np.float64

    # regular grid interpolation with y_1/x_1
    results_regular = gri.interp2grid(data_cf_2dflat, dst_grid, "nearest", try_reg=True)
    results = gri.interp2grid(
        data_cf_2dflat.drop_vars(["lat_1", "lon_1"], errors="ignore"),
        dst_grid,
        "nearest",
        try_reg=False,
    ).transpose(*results_regular.test_var_1.dims)
    xr.testing.assert_identical(
        results_regular.isel(longitude=slice(2, -2), latitude=slice(2, -2)),
        results.isel(longitude=slice(2, -2), latitude=slice(2, -2)),
    )


@pytest.mark.parametrize(
    "interp_kwargs",
    [
        dict(method="nearest", try_reg=True),
        dict(method="nearest", try_reg=False),
        dict(method="nearest_3D"),
        dict(method="linear", try_reg=True),
        dict(method="linear", try_reg=False),
        dict(method="temp", k=25, weight_power=0.75),
        dict(method="temp", k=25, grad_only=True),
        dict(method="temp_fixgrad"),
        dict(method="nearest_alt"),
        dict(method="idw"),
        dict(method="upscaled"),
        dict(method="snowlim"),
    ],
)
def test_interp2poi_preserved_dtype(interp_kwargs, data_native_lv95, poi_def_lv95):
    """
    Test that interp2poi from grid_interp module always return data in the same dtype as input.
    """

    from grid_process.grid_interp import interp2poi

    varname = list(data_native_lv95)[0]

    dtype = np.float32
    assert data_native_lv95[varname].dtype == dtype
    result = interp2poi(data_native_lv95, poi_def_lv95, **interp_kwargs)
    assert result[varname].dtype == dtype

    dtype = np.float64
    result = interp2poi(data_native_lv95.astype(dtype), poi_def_lv95, **interp_kwargs)
    assert result[varname].dtype == dtype


@pytest.mark.parametrize(
    "interp_kwargs",
    [
        dict(method="nearest", try_reg=True),
        dict(method="nearest", try_reg=False),
        dict(method="nearest_3D"),
        dict(method="linear", try_reg=True),
        dict(method="linear", try_reg=False),
        dict(method="temp", k=25, weight_power=0.75),
        dict(method="temp_fixgrad"),
        dict(method="nearest_alt"),
        dict(method="idw"),
    ],
)
def test_interp2grid_preserved_dtype(interp_kwargs, data_native_lv95, data_native_rotlatlon):
    """
    Test that interp2grid from grid_interp module always return data in the same dtype as input.
    """

    from grid_process.grid_interp import interp2grid

    data_native_lv95_shifted = data_native_lv95.copy(deep=True)
    data_native_lv95_shifted = data_native_lv95_shifted.assign_coords(x=data_native_lv95_shifted.x + 100)
    data_native_lv95_shifted = data_native_lv95_shifted.assign_coords(y=data_native_lv95_shifted.y + 100)
    data_native_lv95_shifted = data_native_lv95_shifted.drop_vars(["latitude", "longitude"])
    varname = list(data_native_lv95)[0]

    dtype = np.float32
    assert data_native_lv95[varname].dtype == dtype
    result = interp2grid(data_native_lv95, data_native_rotlatlon, **interp_kwargs)
    assert result[varname].dtype == dtype
    result = interp2grid(data_native_lv95, data_native_lv95_shifted, **interp_kwargs)
    assert result[varname].dtype == dtype

    dtype = np.float64
    data_native_lv95_float64 = data_native_lv95.astype(dtype)
    result = interp2grid(data_native_lv95_float64, data_native_rotlatlon, **interp_kwargs)
    assert result[varname].dtype == dtype
    result = interp2grid(data_native_lv95_float64, data_native_lv95_shifted, **interp_kwargs)
    assert result[varname].dtype == dtype


@pytest.mark.parametrize("method", ["nearest", "linear", "idw"])
def test_interp2grid_integer_coords(method, model_data):
    """
    Test that interp2grid from grid_interp module gives the same results with integer and float
    coordinates of the destination grid: the reprojected coordinates must not be truncated.
    """

    from grid_process.grid_interp import interp2grid

    # unstructured source grid in latitude/longitude
    model_data_flat = model_data.stack({"cell": ("y", "x")}).drop_vars(("cell", "y", "x"))
    # LV95 destination grid within the source domain, with fewer points: reprojected to latitude/longitude
    dst_grid_int = xr.Dataset(
        coords={"x": np.arange(2660000, 2700001, 10000), "y": np.arange(1180000, 1220001, 10000)},
        attrs={"crs": "EPSG:2056"},
    )
    dst_grid = dst_grid_int.assign_coords(x=dst_grid_int.x.astype(np.float64), y=dst_grid_int.y.astype(np.float64))

    expected = interp2grid(model_data_flat, dst_grid, method)
    result = interp2grid(model_data_flat, dst_grid_int, method)
    assert expected.test_var_1.notnull().all()
    np.testing.assert_array_equal(result.test_var_1.values, expected.test_var_1.values)


@pytest.mark.parametrize("method", ["nearest", "idw", "temp_fixgrad"])
def test_interp2poi_integer_coords(method, data_native_lv95, poi_def_lv95):
    """
    Test that interp2poi from grid_interp module gives the same results with integer and float
    spatial coordinates of the source grid: the reprojected POIs coordinates must not be truncated.
    """

    from grid_process.grid_interp import interp2poi

    data_lv95 = data_native_lv95.drop_vars(["latitude", "longitude"])
    data_lv95_int = data_lv95.assign_coords(x=data_lv95.x.astype(np.int64), y=data_lv95.y.astype(np.int64))

    expected = interp2poi(data_lv95, poi_def_lv95, method)
    result = interp2poi(data_lv95_int, poi_def_lv95, method)
    np.testing.assert_array_equal(result.test_var_1.values, expected.test_var_1.values)


# ----------------------------------------------------------------------------------------------------------------------
# CRS of the generic interpolation path (KDTree / griddata): never latitude/longitude degrees
# ----------------------------------------------------------------------------------------------------------------------


def _latlon_grid(lat: np.ndarray, lon: np.ndarray, alt: float = 500.0) -> xr.Dataset:
    """Regular latitude/longitude grid whose values are the flat index of the grid points."""

    shape = (lat.size, lon.size)
    return xr.Dataset(
        {"var": (("latitude", "longitude"), np.arange(lat.size * lon.size, dtype=np.float64).reshape(shape))},
        coords={
            "latitude": lat,
            "longitude": lon,
            "surface_altitude": (("latitude", "longitude"), np.full(shape, alt)),
        },
        attrs={"crs": "EPSG:4326"},
    )


def _latlon_pois(lon: np.ndarray, lat: np.ndarray, alt: float = 500.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": [f"p{i}" for i in range(len(lon))], "height_masl": np.full(len(lon), alt)},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )


@pytest.mark.parametrize(
    "src,dst,src_larger,dst_lonlat,expected",
    [
        # one metric CRS: that one
        ("EPSG:2056", "EPSG:4326", True, None, "EPSG:2056"),
        ("EPSG:4326", "EPSG:2056", True, None, "EPSG:2056"),
        ("EPSG:32632", "EPSG:4326", False, None, "EPSG:32632"),
        # two metric CRS: the one of the larger grid
        ("EPSG:21781", "EPSG:2056", True, None, "EPSG:21781"),
        ("EPSG:21781", "EPSG:2056", False, None, "EPSG:2056"),
        ("EPSG:32632", "EPSG:2056", True, None, "EPSG:32632"),
        # two geographic CRS: LV95 if the destination is close enough to Switzerland, otherwise ECEF
        ("EPSG:4326", "EPSG:4326", True, ([5.9, 10.5], [45.8, 47.8]), "EPSG:2056"),  # Switzerland
        ("EPSG:4326", "EPSG:4326", False, ([-0.8, 17.7], [42.0, 50.5]), "EPSG:2056"),  # ICON-CH1 domain
        ("EPSG:4326", "EPSG:4326", True, ([5.0, 15.0], [55.0, 60.0]), "EPSG:4978"),  # Denmark / Scandinavia
        ("EPSG:4326", "EPSG:4326", True, ([-180.0, 179.0], [-90.0, 90.0]), "EPSG:4978"),  # global
    ],
)
def test_choose_kdtree_crs(src, dst, src_larger, dst_lonlat, expected):
    """
    Choice of the CRS of the nearest neighbor searches and triangulations
    """

    from pyproj import CRS

    import grid_process.grid_interp as gri
    from grid_process.helpers import reproject

    def dst_lv95():
        assert dst_lonlat is not None, "the destination extent is only needed for two geographic CRS"
        lon, lat = np.meshgrid(*dst_lonlat)
        return reproject(lon.ravel(), lat.ravel(), "EPSG:4326", "EPSG:2056")

    target = gri._choose_kdtree_crs(CRS.from_user_input(src), CRS.from_user_input(dst), src_larger, dst_lv95)
    assert target == CRS.from_user_input(expected)


def test_choose_kdtree_crs_threshold(monkeypatch):
    """
    The LV95 range is configurable, and non-finite LV95 coordinates give ECEF
    """

    from pyproj import CRS

    import grid_process.grid_interp as gri
    from grid_process import CONFIG
    from grid_process.helpers import reproject

    latlon = CRS.from_epsg(4326)
    swiss = reproject(np.array([5.9, 10.5]), np.array([45.8, 47.8]), latlon, "EPSG:2056")  # max offset ~ 125 km
    assert gri._choose_kdtree_crs(latlon, latlon, True, lambda: swiss) == gri.LV95
    assert gri._choose_kdtree_crs(latlon, latlon, True, lambda: (np.array([np.inf]), np.array([1.2e6]))) == gri.ECEF
    monkeypatch.setattr(CONFIG, "lv95_max_offset", 50_000.0)
    assert gri._choose_kdtree_crs(latlon, latlon, True, lambda: swiss) == gri.ECEF


def test_interp2poi_ecef_nearest():
    """
    Beyond the LV95 range, nearest neighbors are searched in ECEF coordinates: they are the nearest grid points along
    the Earth's surface (not in latitude/longitude degrees), also across the antimeridian
    """

    from pyproj import Geod

    import grid_process.grid_interp as gri

    def check(grid: xr.Dataset, pois: gpd.GeoDataFrame) -> None:
        result = gri.interp2poi(grid, pois, "nearest", try_reg=False)["var"].values
        lon_2d, lat_2d = np.meshgrid(grid.longitude.values, grid.latitude.values)
        n_poi, n_grid = len(pois), lon_2d.size
        _, _, dist = Geod(ellps="WGS84").inv(
            np.repeat(pois.geometry.x.values, n_grid),
            np.repeat(pois.geometry.y.values, n_grid),
            np.tile(lon_2d.ravel(), n_poi),
            np.tile(lat_2d.ravel(), n_poi),
        )
        dist = dist.reshape(n_poi, n_grid)
        expected = np.argmin(dist, axis=1)
        # ties (nearly equidistant grid points) may be resolved differently between chord and geodesic distances
        sorted_dist = np.sort(dist, axis=1)
        tie = sorted_dist[:, 1] - sorted_dist[:, 0] < 1e-6 * sorted_dist[:, 1]
        np.testing.assert_array_equal(result[~tie], expected[~tie])

    rng = np.random.default_rng(1)
    # Europe, up to 70°N, where 1° of longitude is about a third of 1° of latitude
    grid = _latlon_grid(np.arange(30.0, 70.1, 1.0), np.arange(-20.0, 40.1, 1.0))
    check(grid, _latlon_pois(rng.uniform(-19, 39, 300), rng.uniform(31, 69, 300)))

    # global grid: POIs close to the antimeridian (the POIs span all longitudes, so that the grid is not sliced)
    grid = _latlon_grid(np.arange(-80.0, 80.1, 2.0), np.arange(-180.0, 179.0, 2.0))
    lon = np.array([179.6, -179.6, 178.9, -178.9, 0.0])
    lat = np.array([0.3, -10.2, 60.4, -61.0, 0.0])
    check(grid, _latlon_pois(lon, lat))
    result = gri.interp2poi(grid, _latlon_pois(lon, lat), "nearest", try_reg=False)["var"].values
    # e.g. 179.6°E is closer to 180°W (index 0 along longitude) than to 178°E
    assert result[0] == grid["var"].sel(latitude=0.0, longitude=-180.0).item()


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("nearest", {}),
        ("idw", {"k": 4, "weight_power": 2}),
        ("upscaled", {"k": 4}),
        ("nearest_3D", {}),
        ("idw_3D", {"k": 4}),
        ("upscaled_3D", {"k": 4}),
        ("temp", {"k": 20, "weight_power": 0.5}),
        ("temp_fixgrad", {"r_search": 1.5}),
        ("nearest_alt", {"r_search": 1.5}),
        ("snowlim", {"k": 9}),
    ],
)
def test_interp2poi_ecef_vs_lv95(method, kwargs, model_data, poi_def, monkeypatch):
    """
    On a Swiss grid, forcing ECEF coordinates gives (almost) the same results as LV95 for all KDTree methods,
    including the methods with altitude (altitude as last column of the coordinates)
    """

    import grid_process.grid_interp as gri
    from grid_process import CONFIG

    lv95 = gri.interp2poi(model_data, poi_def, method, try_reg=False, **kwargs)
    monkeypatch.setattr(CONFIG, "lv95_max_offset", 1.0)
    ecef = gri.interp2poi(model_data, poi_def, method, try_reg=False, **kwargs)
    # LV95 (projection scale error) and ECEF (chord) distances differ slightly, which changes the inverse distance
    # weights by ~1e-6 of the data range; absolute tolerance as "temp" can extrapolate values close to 0
    for var in lv95.data_vars:
        atol = 1e-5 * float(np.ptp(model_data[var].values))
        np.testing.assert_allclose(lv95[var].values, ecef[var].values, rtol=1e-4, atol=atol, err_msg=var)


@pytest.mark.parametrize("method", ["linear", "linear_3D", "cubic"])
def test_griddata_ecef_error(method, model_data):
    """
    griddata methods (Delaunay triangulation) raise an error beyond the LV95 range (ECEF coordinates)
    """

    import grid_process.grid_interp as gri

    grid = _latlon_grid(np.arange(55.0, 70.1, 0.5), np.arange(5.0, 25.1, 0.5))
    pois = _latlon_pois(np.array([10.0, 20.0]), np.array([60.0, 65.0]))
    with pytest.raises(ValueError, match="earth-centered earth-fixed"):
        gri.interp2poi(grid, pois, method, try_reg=False)
    dst_grid = _latlon_grid(np.arange(56.0, 69.1, 1.0), np.arange(6.0, 24.1, 1.0)).drop_vars("var")
    with pytest.raises(ValueError, match="earth-centered earth-fixed"):
        gri.interp2grid(grid, dst_grid, method, try_reg=False)


def test_interp2grid_unstructured_dst_error(model_data):
    """
    The destination grid of interp2grid must be structured (two spatial dimensions)
    """

    import grid_process.grid_interp as gri

    dst_flat = model_data.stack({"cell": ("y", "x")}).drop_vars(("cell", "y", "x"))
    with pytest.raises(ValueError, match="structured grid"):
        gri.interp2grid(model_data, dst_flat, "nearest")


def test_kdtree_native_metric_coords(data_native_lv95, poi_def_lv95):
    """
    Native coordinates in meters (here LV03) are used instead of the latitude/longitude coordinates, and a grid
    with only latitude/longitude coordinates gives the same nearest neighbors as with its native LV95 coordinates
    """

    import grid_process.grid_interp as gri
    from grid_process.ds_utils import ds_to_da

    # LV03 grid with latitude/longitude coordinates: nearest neighbor search in LV03
    data_lv03 = data_native_lv95.assign_coords(x=data_native_lv95.x - 2e6, y=data_native_lv95.y - 1e6)
    data_lv03.attrs["grid_mapping"] = '{"epsg_code": "EPSG:21781"}'
    da, _ = ds_to_da(data_lv03)
    coords_flat, _, pois_coords = gri._flatten_concat(da.transpose("y", "x", ...), ("y", "x"), poi_def_lv95, "idw")
    for coords in (coords_flat, pois_coords):
        assert (coords[:, 0] > 5e5).all() and (coords[:, 0] < 9e5).all()  # LV03 easting
        assert (coords[:, 1] > 5e4).all() and (coords[:, 1] < 3.5e5).all()  # LV03 northing

    # the same LV95 grid with only latitude/longitude coordinates
    data_latlon = data_native_lv95.drop_vars(["x", "y"]).assign_attrs(crs="EPSG:4326")
    data_latlon.attrs.pop("grid_mapping")
    for method, kwargs in (("nearest", {}), ("idw", {"k": 4, "weight_power": 2})):
        native = gri.interp2poi(data_native_lv95, poi_def_lv95, method, try_reg=False, **kwargs)
        latlon = gri.interp2poi(data_latlon, poi_def_lv95, method, try_reg=False, **kwargs)
        xr.testing.assert_allclose(native.drop_vars(["x", "y"], errors="ignore"), latlon, rtol=1e-4)


def _sphere_field(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Smooth field on the sphere, continuous across the 360 degree seam and at the poles"""

    lon, lat = np.radians(lon), np.radians(lat)
    return 10 * np.cos(lat) * np.cos(lon) + 5 * np.sin(lat)


def _global_grid(res: float, lon_start: float, lon_span: float = 360.0, lat_bnds=(90.0, -90.0), var=True):
    """Regular latitude/longitude grid (decreasing latitudes, as ERA5 or GFS) with the field of _sphere_field"""

    lat = np.arange(lat_bnds[0], lat_bnds[1] - res / 2, -res)
    lon = np.arange(lon_start, lon_start + lon_span - res / 2, res)
    grid = xr.Dataset(coords={"latitude": lat, "longitude": lon}, attrs={"crs": "EPSG:4326"})
    if var:
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        grid["var"] = (("latitude", "longitude"), _sphere_field(lon_2d, lat_2d).astype(np.float32))
    return grid


@pytest.mark.parametrize(
    "method,kwargs", [("nearest", {}), ("linear", {}), ("nearest", {"try_reg": False}), ("idw", {})]
)
@pytest.mark.parametrize(
    "src_lon,dst_lon,dst_span,dst_lat",
    [
        (0.0, -180.0, 360.0, (90.0, -90.0)),  # 0..360 to -180..180
        (-180.0, 0.0, 360.0, (90.0, -90.0)),  # -180..180 to 0..360
        (-180.0, 170.0, 20.0, (30.0, -30.0)),  # destination across the antimeridian, 170..190
        (0.0, -190.0, 20.0, (30.0, -30.0)),  # destination across the source seam, -190..-170
    ],
)
def test_interp2grid_longitude_convention(method, kwargs, src_lon, dst_lon, dst_span, dst_lat):
    """
    Latitude/longitude source and destination grids with different longitude conventions (0..360 or -180..180):
    all destination cells get the value of their location, including across the 360 degree seam
    """

    import grid_process.grid_interp as gri

    src = _global_grid(1.0, src_lon)
    dst = _global_grid(0.5, dst_lon, dst_span, dst_lat, var=False)
    out = gri.interp2grid(src, dst, method, **kwargs)["var"].transpose("latitude", "longitude").values
    expected = _sphere_field(*np.meshgrid(dst.longitude.values, dst.latitude.values))
    assert not np.isnan(out).any()
    # max gradient of the field ~0.2 per degree: nearest is within half a grid spacing
    np.testing.assert_allclose(out, expected, atol=0.01 if method == "linear" else 0.15)


@pytest.mark.parametrize("src_lon", [0.0, -180.0])
def test_interp2poi_longitude_convention(src_lon):
    """
    POIs at any longitude (incl. the 360 degree seam and near the poles) on a global latitude/longitude grid in
    either longitude convention, through the regular path (try_reg=True) and the generic (ECEF) path
    """

    import grid_process.grid_interp as gri

    src = _global_grid(1.0, src_lon)
    rng = np.random.default_rng(0)
    src = src.assign_coords(surface_altitude=(("latitude", "longitude"), rng.uniform(0, 3000, src["var"].shape)))
    lon = np.array([-179.9, 179.9, 359.6, -70.0, 0.0, 0.2, 139.7, -0.1, 290.0, -120.0])
    lat = np.array([-18.0, 0.0, 10.0, -16.5, -89.99, 89.9, 35.7, 51.5, 60.0, 40.0])
    pois = gpd.GeoDataFrame(
        {"name": [f"p{i}" for i in range(lon.size)], "height_masl": 500.0},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    expected = _sphere_field(lon, lat)
    for method, kwargs, atol in [
        ("nearest", {}, 0.15),
        ("linear", {}, 0.01),
        ("nearest", {"try_reg": False}, 0.15),
        ("temp", {"k": 9}, 0.5),
    ]:
        out = gri.interp2poi(src, pois, method, **kwargs)["var"].values
        np.testing.assert_allclose(out, expected, atol=atol, err_msg=f"{method} {kwargs}")


def test_interp2grid_regional_latlon_source_unchanged():
    """
    A regional latitude/longitude source is not wrapped around the globe: the regular path gives NaN outside of
    its domain and the expected values inside
    """

    import grid_process.grid_interp as gri

    src = _global_grid(1.0, -10.0, 41.0, (70.0, 30.0))
    dst = _global_grid(0.5, -180.0, 360.0, (80.0, 20.0), var=False)
    out = gri.interp2grid(src, dst, "linear")["var"].transpose("latitude", "longitude").values
    lon_2d, lat_2d = np.meshgrid(dst.longitude.values, dst.latitude.values)
    inside = (lon_2d >= -10) & (lon_2d <= 30) & (lat_2d <= 70) & (lat_2d >= 30)
    assert np.isnan(out[~inside]).all()
    np.testing.assert_allclose(out[inside], _sphere_field(lon_2d, lat_2d)[inside], atol=0.01)


def test_interp2poi_temp_at_poles():
    """
    On a latitude/longitude grid with rows at the poles, all points of a pole row are copies of the same point
    (same altitude and value): the k nearest neighbors of a POI close to a pole may all be such copies, for which
    no sounding can be fitted. temp falls back to the nearest grid point corrected with a fixed lapse rate
    """

    import grid_process.grid_interp as gri

    src = _global_grid(1.0, 0.0)
    rng = np.random.default_rng(0)
    alt = rng.uniform(0, 3000, src["var"].shape)
    alt[[0, -1], :] = alt[[0, -1], :1]  # one altitude per pole
    src = src.assign_coords(surface_altitude=(("latitude", "longitude"), alt))
    lon, lat = np.array([0.0, 120.0, -45.0]), np.array([-90.0, -89.99, 89.95])
    pois = gpd.GeoDataFrame(
        {"name": ["south_pole", "near_south_pole", "near_north_pole"], "height_masl": alt[[-1, -1, 0], 0]},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    out = gri.interp2poi(src, pois, "temp", z_fact=15, k=50, weight_power=0.75)["var"].values
    # same altitude as the pole point: its value
    np.testing.assert_allclose(out, _sphere_field(lon, lat), atol=0.01)


def _flatten_test_grids() -> dict[str, tuple[xr.DataArray, tuple[str, ...]]]:
    """
    Grids with the spatial dimensions first: 1D dimension coordinates, 2D coordinates stored in another dimension
    order (x, y), unstructured
    """

    rng = np.random.default_rng(0)
    lat = np.linspace(50, 40, 5)
    lon = np.linspace(0, 10, 7)
    grid_1d = xr.DataArray(
        rng.standard_normal((5, 7, 3)),
        dims=("latitude", "longitude", "step"),
        coords={"latitude": lat, "longitude": lon},
    )
    lat_xy = rng.uniform(40, 50, (7, 5))
    grid_2d = xr.DataArray(
        rng.standard_normal((5, 7, 2, 3)).astype(np.float32),
        dims=("y", "x", "step", "realization"),
        coords={
            "y": np.arange(5.0),
            "x": np.arange(7.0),
            "latitude": (("x", "y"), lat_xy),
            "longitude": (("x", "y"), lat_xy + 1),
            "surface_altitude": (("y", "x"), rng.uniform(0, 3000, (5, 7))),
        },
    )
    unstructured = xr.DataArray(
        rng.standard_normal((9, 2)),
        dims=("cell", "step"),
        coords={"lat": ("cell", rng.uniform(40, 50, 9)), "lon": ("cell", rng.uniform(0, 10, 9))},
    )
    return {
        "1d_coords": (grid_1d, ("latitude", "longitude")),
        "2d_coords_xy": (grid_2d, ("y", "x")),
        "unstructured": (unstructured, ("cell",)),
    }


@pytest.mark.parametrize("case", ["1d_coords", "2d_coords_xy", "unstructured"])
def test_flat_coord_values(case):
    """
    _flat_coord and _flat_values give the same values and order as xarray.stack
    """

    from grid_process.grid_interp import _flat_coord, _flat_values  # pylint: disable=import-outside-toplevel

    grid, spatial_dims = _flatten_test_grids()[case]
    stacked = grid.stack({"flat": spatial_dims}).transpose("flat", ...) if len(spatial_dims) > 1 else grid
    for name in (name for name, coord in grid.coords.items() if set(coord.dims) <= set(spatial_dims)):
        np.testing.assert_array_equal(_flat_coord(grid, name, spatial_dims), stacked[name].values)
    flat = _flat_values(grid, spatial_dims)
    np.testing.assert_array_equal(flat, stacked.values)
    assert flat.dtype == grid.dtype

    with pytest.raises(ValueError, match="other than the spatial dimensions"):
        _flat_coord(grid.assign_coords(bad=("step", np.arange(grid.sizes["step"]))), "bad", spatial_dims)


def test_valid_points():
    """
    _valid_points excludes only the points that are NaN (or NaT) in all fields, and gives None when no copy is needed
    """

    from grid_process.grid_interp import _valid_points  # pylint: disable=import-outside-toplevel

    data = np.arange(12.0).reshape(4, 3)
    assert _valid_points(data) is None
    data[1, 0] = np.nan  # NaN in some fields only: kept
    assert _valid_points(data) is None
    data[2] = np.nan  # NaN in all fields: excluded
    data[3, 1:] = np.nan  # NaN in all fields but the first (NaN in the middle one, checked further): kept
    np.testing.assert_array_equal(_valid_points(data), [True, True, False, True])
    np.testing.assert_array_equal(_valid_points(data.reshape(4, 3, 1)), [True, True, False, True])
    np.testing.assert_array_equal(_valid_points(data[:, 0]), [True, False, False, True])
    data[0, [0, 2]] = np.nan  # NaN in all fields but the middle one: kept
    np.testing.assert_array_equal(_valid_points(data), [True, True, False, True])

    # accumulated variable: first lead time NaN everywhere, some points masked (NaN in all lead times)
    accum = np.arange(20.0).reshape(4, 5)
    accum[:, 0] = np.nan
    assert _valid_points(accum) is None
    accum[1] = np.nan
    np.testing.assert_array_equal(_valid_points(accum), [True, False, True, True])
    np.testing.assert_array_equal(_valid_points(accum.reshape(4, 1, 5)), [True, False, True, True])

    assert _valid_points(np.arange(6).reshape(3, 2)) is None  # integers cannot be NaN
    times = np.array([["2026-01-01", "NaT"], ["NaT", "NaT"]], dtype="datetime64[ns]")
    np.testing.assert_array_equal(_valid_points(times), [True, False])
    # no field: all() of an empty selection is True, as with xarray
    np.testing.assert_array_equal(_valid_points(np.empty((3, 0))), [False, False, False])

    # same as the previous xarray implementation
    da = xr.DataArray(data, dims=("cell", "step"))
    np.testing.assert_array_equal(_valid_points(data), ~da.isnull().all("step").values)
    rng = np.random.default_rng(0)
    scattered = rng.random((200, 3, 4))
    scattered[rng.random(scattered.shape) < 0.8] = np.nan
    scattered[:20] = np.nan
    expected = ~xr.DataArray(scattered, dims=("cell", "member", "step")).isnull().all(("member", "step")).values
    assert not expected.all()
    np.testing.assert_array_equal(_valid_points(scattered), expected)


def test_interp2poi_all_pois_outside_unstructured_grid():
    """
    POIs all outside an unstructured grid (no grid point left after slicing) get NaN, with the other dimensions,
    without numpy warnings
    """

    import grid_process.grid_interp as gri  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(0)
    data = xr.Dataset(
        {"t": (("step", "cell"), rng.random((2, 50), dtype=np.float32))},
        coords={
            "step": [0, 1],
            "latitude": ("cell", rng.uniform(46, 47, 50)),
            "longitude": ("cell", rng.uniform(7, 8, 50)),
        },
        attrs={"crs": "EPSG:4326"},
    )
    pois = gpd.GeoDataFrame({"name": ["paris"]}, geometry=gpd.points_from_xy([2.35], [48.85]), crs="EPSG:4326")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # e.g. "Mean of empty slice"
        out = gri.interp2poi(data, pois, "nearest")

    assert out["t"].dims == ("poi", "step")
    assert out["t"].dtype == np.float32
    np.testing.assert_array_equal(out["step"], [0, 1])
    assert out["t"].isnull().all()


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        ("nearest", {}, 1),
        ("linear", {}, 1),
        ("cubic", {}, 2),
        ("idw_3D", {"k": 4}, 4),
        ("idw", {"k": 9, "weight_power": 0.25}, 3),
        ("nearest_alt", {"r_search": 1.7}, 3),
        ("temp", {"z_fact": 15, "k": 50, "weight_power": 0.75}, 5),
    ],
)
def test_get_buffer(method, kwargs, expected):
    """
    The slicing buffer grows with the number of neighbors ``k``, the search radius ``r_search``, and for ``cubic``
    and the 3D methods
    """

    from grid_process.grid_interp import get_buffer  # pylint: disable=import-outside-toplevel

    assert get_buffer(method, **kwargs) == expected


def test_window_nanmean():
    """
    Coarsening skips NaN (a block is NaN only if all its values are), keeps float32, and leaves integers to np.mean
    """

    from grid_process.grid_interp import _window_nanmean  # pylint: disable=import-outside-toplevel

    values = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    values[0, 0, 1] = np.nan  # block (0, 0) of the first field: 3 valid values
    values[1, 2:, 2:] = np.nan  # block (1, 1) of the second field: all NaN
    da = xr.DataArray(values, dims=("step", "y", "x"))

    out = da.coarsen(y=2, x=2, boundary="trim").reduce(_window_nanmean)

    assert out.dtype == np.float32
    with warnings.catch_warnings():  # np.nanmean warns for the all-NaN block, bottleneck does not
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanmean(values.reshape(2, 2, 2, 2, 2), axis=(2, 4))
    np.testing.assert_allclose(out.values, expected, rtol=1e-6)
    assert out.values[0, 0, 0] == np.float32((0 + 4 + 5) / 3)
    assert np.isnan(out.values[1, 1, 1])
    assert np.isnan(out.values).sum() == 1

    ints = xr.DataArray(np.arange(16, dtype=np.int16).reshape(4, 4), dims=("y", "x"))
    np.testing.assert_array_equal(
        ints.coarsen(y=2, x=2).reduce(_window_nanmean), ints.coarsen(y=2, x=2).reduce(np.mean)
    )


def test_window_nanmean_large_window():
    """Large float32 windows with NaN: close to the float64 mean (float32 sums), all-NaN blocks stay NaN"""

    from grid_process.grid_interp import _window_nanmean  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(0)
    values = (rng.standard_normal((3, 2, 20, 30)) * 10 + 280).astype(np.float32)
    values[rng.random(values.shape) < 0.3] = np.nan
    values[0, 0, :10, :10] = np.nan  # an all-NaN block
    da = xr.DataArray(values, dims=("step", "member", "y", "x"))

    out = da.coarsen(y=10, x=10).reduce(_window_nanmean)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanmean(values.astype(np.float64).reshape(3, 2, 2, 10, 3, 10), axis=(3, 5))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.values, expected, rtol=1e-6)
    assert np.isnan(out.values[0, 0, 0, 0])
    assert np.isnan(out.values).sum() == 1
