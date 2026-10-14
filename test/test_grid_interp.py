# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the grid_interp module"""

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

    # ***flat orography: all neighbor altitudes identical → deg=1 rank-deficient***
    # must raise ValueError without infinite recursion

    x_flat = np.tile(np.arange(0, 10), (10, 1))
    y_flat = x_flat.T
    z_flat = np.zeros(x_flat.shape)  # completely flat
    coords_flat_flat = np.c_[x_flat.flatten(), y_flat.flatten(), z_flat.flatten()]
    data_flat_flat = (x_flat + y_flat).flatten().astype(np.float32)[:, None]
    poi_coords_flat = np.array([[2.5, 2.5, 0.0], [7.0, 7.0, 0.0]])

    with pytest.raises(ValueError, match="rank-deficient"):
        gri.interp_temp(coords_flat_flat, data_flat_flat, poi_coords_flat, z_fact=1, k=9)

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
    func1 = lambda x: 2 * np.power(x, 3) + 3 * x - 2
    func2 = lambda x: 2 * x + 4
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

        if method in methods_2:
            # coords transformed to 'EPSG:2056' (CH1903+/LV95)
            assert (pois_coords[:, :2] > 1e5).all() and (pois_coords[:, :2] < 1e7).all()
            assert (model_coords_flat[:, :2] > 1e5).all() and (model_coords_flat[:, :2] < 1e7).all()
        else:
            # coords still in 'EPSG:4326' (lat/lon)
            assert (pois_coords[:, :2] > -180).all() and (pois_coords[:, :2] < 180).all()
            assert (model_coords_flat[:, :2] > -180).all() and (model_coords_flat[:, :2] < 180).all()


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

    # check that with a single POI the sliced grid has correct sizes
    sliced_data = gri.slice_boundingbox(model_data_2dflat, poi_def.loc[[0]], method="nearest")
    if "cell" in sliced_data.dims:
        assert 4 <= sliced_data.sizes["cell"] <= 16
    else:
        assert sliced_data.sizes["x"] == 2
        assert sliced_data.sizes["y"] == 2

    sliced_data = gri.slice_boundingbox(data_native_lv95_2dflat, poi_def_lv95.loc[[0]], method="nearest")
    if "cell" in sliced_data.dims:
        assert 4 <= sliced_data.sizes["cell"] <= 16
    else:
        assert sliced_data.sizes["x"] == 2
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
        xr.testing.assert_identical(results_regular, results)

        # regular grid interpolation with y/x and y/x for dst_grid
        results_regular_2 = gri.interp2grid(
            model_data_2dflat,
            data_native_lv95.drop_vars(["longitude", "latitude"]),
            method,
            try_reg=True,
        )

        if method == "linear" and "cell" in model_data_2dflat.dims:
            results_regular_2.update(results.drop_vars(["longitude", "latitude"]))

        xr.testing.assert_identical(results_regular_2, results.drop_vars(["longitude", "latitude"]))

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
        xr.testing.assert_allclose(results.where(results_regular.notnull()), results_regular)

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
