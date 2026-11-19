# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ens_stats module"""

import warnings

import pytest
import numpy as np
import xarray as xr
import grid_process.ens_stats as ens


def test_ens_prob(dummy_members):
    """Simple testcase for the ens_prob function from ens_stats module"""

    n_members = len(dummy_members.realization)
    thresh_above = [2, 4]
    probname_above = ["prob_above_2mm", "prob_above_4mm"]
    results_above = ens.ens_prob(dummy_members, thresh_above, probname_above, "above")

    assert list(results_above) == probname_above
    assert np.all([da.units == "percent" for var, da in results_above.items()])
    assert np.allclose(results_above["prob_above_2mm"], (n_members - 2) / n_members * 100)
    assert np.allclose(results_above["prob_above_4mm"], (n_members - 4) / n_members * 100)

    thresh_below = 3
    probname_below = "prob_below_3mm"
    results_below = ens.ens_prob(dummy_members, thresh_below, probname_below, "below")
    assert list(results_below) == [probname_below]
    assert np.all([da.units == "percent" for var, da in results_below.items()])
    assert np.allclose(results_below["prob_below_3mm"], 3 / n_members * 100)


def test_agg_prob(model_data_validtime):
    """Simple testcase for the agg_prob function from ens_stats module"""

    member_count = [3, 5]
    n_members = len(model_data_validtime.realization)
    dict_prob = dict(
        prob_names=["prob_0_24h", "prob_1_24h"],
        thresholds=[np.sum(np.arange(24)) + n * 24 for n in member_count],
        granularity=24,
        interval=24,
        offset=0,
    )

    results = ens.agg_prob(
        model_data_validtime,
        dict_prob=dict_prob,
        agg_func=lambda x: x.sum("window"),
        sign="below",
    )

    results_lt_duplicates_false = ens.agg_prob(
        model_data_validtime,
        dict_prob=dict_prob,
        agg_func=lambda x: x.sum("window"),
        lt_duplicates=False,
        sign="below",
    )

    assert list(results) == dict_prob["prob_names"]
    assert np.all([da.units == "percent" for var, da in results.items()])
    assert np.allclose(results["prob_0_24h"], member_count[0] / n_members * 100)
    assert np.allclose(results["prob_1_24h"], member_count[1] / n_members * 100)
    assert len(results.time) == 8
    assert len(results_lt_duplicates_false.time) == 4


def test_neighbor_stat(dummy_members):
    """Simple testcase for the neighbor_stat function from ens_stats module"""

    w_size_xy = 3
    half_w = int(w_size_xy / 2)
    results_mean = ens.neighbor_stat(dummy_members, method="mean", w_size_xy=w_size_xy)

    results_min = ens.neighbor_stat(dummy_members, method="min", w_size_xy=w_size_xy, w_size_t=w_size_xy + 2)

    results_sum = ens.neighbor_stat(dummy_members, method="sum", w_size_xy=w_size_xy, w_size_t=None)

    results_max_timeonly = ens.neighbor_stat(dummy_members, method="max", w_size_xy=None, w_size_t=w_size_xy)

    assert (
        dummy_members.sizes
        == results_mean.sizes
        == results_sum.sizes
        == results_min.sizes
        == results_max_timeonly.sizes
    )

    expected_sum = dummy_members.realization * w_size_xy**2
    expected_sum_borders = dummy_members.realization * w_size_xy * (w_size_xy - half_w)
    expected_sum_corners = dummy_members.realization * (w_size_xy - half_w) ** 2
    results_mean = results_mean.to_array()
    results_min = results_min.to_array()
    results_sum = results_sum.to_array()
    results_max_timeonly = results_max_timeonly.to_array()

    assert np.allclose(dummy_members.to_array(), results_mean)
    assert np.allclose(results_mean, results_min)
    assert np.allclose(results_mean, results_max_timeonly)

    assert np.allclose(
        *xr.broadcast(
            results_sum.isel(x=slice(half_w, -half_w), y=slice(half_w, -half_w)),
            expected_sum,
        )
    )

    arrays2compare_borders = xr.broadcast(
        results_sum.isel(x=0, y=slice(1, -1)),
        results_sum.isel(x=-1, y=slice(1, -1)),
        results_sum.isel(x=slice(1, -1), y=0),
        results_sum.isel(x=slice(1, -1), y=-1),
        expected_sum_borders,
    )

    assert all([np.allclose(arrays2compare_borders[0], da) for da in arrays2compare_borders[1:]])

    arrays2compare_corners = xr.broadcast(
        results_sum.isel(x=0, y=0),
        results_sum.isel(x=-1, y=0),
        results_sum.isel(x=0, y=-1),
        results_sum.isel(x=-1, y=-1),
        expected_sum_corners,
    )

    assert all([np.allclose(arrays2compare_corners[0], da) for da in arrays2compare_corners[1:]])

    with pytest.raises(ValueError, match="method must be one of"):
        ens.neighbor_stat(dummy_members, method="median", w_size_t=5)


def test_upscaled_prob(dummy_members):
    """Simple testcase for the upscaled_prob function from ens_stats module"""

    w_size_xy = 3
    half_w = int(w_size_xy / 2)
    y_mid = int(len(dummy_members.y) / 2)
    x_mid = int(len(dummy_members.x) / 2)
    n_members = len(dummy_members.realization)
    thresh = [2, 4]
    probname = ["prob_upscaled>2mm", "prob_upscaled>4mm"]
    data_test = dummy_members + 0
    for var in dummy_members:
        data_test[var][dict(y=y_mid, x=x_mid)] += 2

    results = ens.upscaled_prob(data_test, thresh, probname, w_size_xy=w_size_xy, w_size_t=w_size_xy + 1)

    assert list(results) == probname
    assert np.all([da.units == "percent" for var, da in results.items()])

    indexer_mid = dict(
        y=slice(y_mid - half_w, y_mid + half_w + 1),
        x=slice(x_mid - half_w, x_mid + half_w + 1),
    )
    indexer_edge = dict(y=slice(0, y_mid - half_w), x=slice(0, x_mid - half_w))
    assert np.allclose(results["prob_upscaled>2mm"][indexer_mid], (n_members - 2 + 2) / n_members * 100)
    assert np.allclose(results["prob_upscaled>4mm"][indexer_mid], (n_members - 4 + 2) / n_members * 100)
    assert np.allclose(results["prob_upscaled>2mm"][indexer_edge], (n_members - 2) / n_members * 100)
    assert np.allclose(results["prob_upscaled>4mm"][indexer_edge], (n_members - 4) / n_members * 100)


def test_circular_mean(model_data):
    """Simple testcase for the ciruclar_mean function from ens_stats module"""

    model_data = model_data % 360

    results = ens.circular_mean(model_data)
    assert isinstance(results, xr.Dataset)
    assert np.all(results.to_array() >= 0) and np.all(results.to_array() < 360)
    assert "realization" not in results.dims

    model_data_dims = {key: value for key, value in model_data.sizes.items() if key != "realization"}
    assert model_data_dims == results.sizes

    results = ens.circular_mean(xr.Dataset({"direction": (["realization"], [359, 1])}))
    assert int(results.direction) == 0


def test_circular_quantile(model_data):
    """
    Simple testcase for the circular_quantile function from ens_stats module
    """

    model_data = model_data % 360

    q = [0.1, 0.5, 0.9]
    results = ens.circular_quantile(model_data, q)
    assert isinstance(results, xr.Dataset)
    assert np.all(results.to_array() >= 0) and np.all(results.to_array() < 360)
    assert list(results["quantile"].values) == q
    assert "realization" not in results.dims

    model_data_dims = {key: value for key, value in model_data.sizes.items() if key != "realization"}
    results_dims = {key: value for key, value in results.sizes.items() if key != "quantile"}
    assert model_data_dims == results_dims

    results = ens.circular_quantile(xr.Dataset({"direction": (["realization"], [359, 1])}), q=0.5, wrap_around=True)
    assert int(results.direction[0]) == 0

    results = ens.circular_quantile(xr.Dataset({"direction": (["realization"], [359, 1])}), q=0.5, wrap_around=False)
    assert int(results.direction[0]) == 360


def test_extended_datamodel(model_data, model_data_validtime):
    """
    Simple testcase for the ens_stats module functions run on extended data model
    """

    from grid_process.data_model import LEADTIME, VALIDTIME, ENS_DIM

    for ensdim in ENS_DIM:
        for leadtime in LEADTIME:
            for validtime in VALIDTIME:
                ds_ext = model_data.rename(realization=ensdim, lead_time=leadtime, time=validtime)
                ds_ext_validtime = model_data_validtime.rename(realization=ensdim, time=validtime)

                dict_prob = dict(
                    prob_names=["prob_2_24h"],
                    thresholds=[2],
                    granularity=24,
                    interval=24,
                    offset=0,
                )
                ds_out = ens.agg_prob(
                    ds_ext_validtime,
                    dict_prob=dict_prob,
                    agg_func=lambda x: x.sum("window"),
                    sign="below",
                )
                assert validtime in ds_out.dims
                assert ensdim not in ds_out.coords

                ds_out = ens.neighbor_stat(ds_ext, method="mean", w_size_xy=3, w_size_t=3)
                assert leadtime in ds_out.dims
                assert validtime in ds_out.coords
                ds_out = ens.neighbor_stat(ds_ext.rename(x="lon", y="lat"), method="mean", w_size_xy=5)
                assert leadtime in ds_out.dims
                assert validtime in ds_out.coords
                assert "lon" in ds_out.dims
                assert "lat" in ds_out.dims
                ds_out = ens.neighbor_stat(ds_ext_validtime, method="max", w_size_xy=3, w_size_t=5)
                assert validtime in ds_out.dims

                ds_out = ens.upscaled_prob(ds_ext, 2, "dummy_prob", w_size_xy=3, w_size_t=4)
                assert leadtime in ds_out.dims
                assert validtime in ds_out.coords
                assert ensdim not in ds_out.coords
                ds_out = ens.upscaled_prob(
                    ds_ext.rename(x="longitude", y="latitude"),
                    1,
                    "dummy_prob2",
                    w_size_xy=5,
                )
                assert leadtime in ds_out.dims
                assert validtime in ds_out.coords
                assert "longitude" in ds_out.dims
                assert "latitude" in ds_out.dims
                assert ensdim not in ds_out.coords
                ds_out = ens.upscaled_prob(ds_ext_validtime, 3, "dummy_prob3", w_size_t=5)
                assert validtime in ds_out.dims
                assert ensdim not in ds_out.coords

        ds_out = ens.ens_prob(ds_ext, 5, "dummy_prob")
        assert ensdim not in ds_out.dims
        assert ensdim not in ds_out.coords

        ds_out = ens.circular_mean(ds_ext_validtime)
        assert ensdim not in ds_out.dims
        assert ensdim not in ds_out.coords

        ds_out = ens.circular_quantile(ds_ext_validtime, q=[0.25, 0.5, 0.75])
        assert ensdim not in ds_out.dims
        assert ensdim not in ds_out.coords
        assert "quantile" in ds_out.dims
        assert "quantile" in ds_out.coords


def _grid_time(dtype=np.float32, nan: bool = True) -> xr.Dataset:
    """Regular grid with 6 lead times of random values, 2 data variables (one without the time dimension)"""

    rng = np.random.default_rng(2)
    values = (rng.standard_normal((6, 9, 11)) * 10).astype(dtype)
    if nan:
        values[rng.random(values.shape) < 0.3] = np.nan
        values[:, 2:6, 3:8] = np.nan  # all-NaN block in space and time
    return xr.Dataset(
        {"var": (("lead_time", "y", "x"), values), "static": (("y", "x"), values[0])},
        coords={"x": np.arange(11.0), "y": np.arange(9.0), "lead_time": np.arange(6)},
        attrs={"source": "test"},
    )


def _rolling(ds: xr.Dataset, method: str, w_size_xy: int | None, w_size_t: int | None) -> xr.Dataset:
    """The former implementation: rolling along x, y then time"""

    for dim, w_size in (("x", w_size_xy), ("y", w_size_xy), ("lead_time", w_size_t)):
        if w_size is not None:
            ds = getattr(ds.rolling({dim: w_size}, min_periods=1, center=True), method)(skipna=False)
    return ds


@pytest.mark.parametrize("method", ["sum", "min", "max"])
@pytest.mark.parametrize("w_size_xy,w_size_t", [(2, None), (3, None), (4, 2), (5, 3), (None, 4), (17, 3)])
def test_neighbor_stat_regular_as_rolling(method, w_size_xy, w_size_t):
    """sum, min and max equal the former rolling along each dimension (odd and even windows, truncated at the edges)"""

    for nan in (False, True):
        grid = _grid_time(nan=nan)
        result = ens.neighbor_stat(grid, method, w_size_xy=w_size_xy, w_size_t=w_size_t)
        expected = _rolling(grid, method, w_size_xy, w_size_t)
        assert result["var"].dtype == np.float32
        assert result.attrs == grid.attrs
        if method == "sum":
            xr.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-4)
        else:
            xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("structure", ["regular", "unstructured"])
@pytest.mark.parametrize("method,fill", [("max", -np.inf), ("min", np.inf)])
def test_neighbor_stat_extreme_infinite(structure, method, fill):
    """A window of only NaN is NaN, a window of only infinite values (the fill value of min/max) stays infinite"""

    grid = _grid_lv95(nan=False)
    values = grid["var"].values
    values[0, :, :6] = np.nan
    values[0, :, 9:] = fill
    if structure == "unstructured":
        result = ens.neighbor_stat(_flat_shuffled(grid), method, w_size_xy=3)
        result = result.set_index(cell=("y", "x")).unstack("cell").transpose(*grid["var"].dims).reindex_like(grid)
    else:
        result = ens.neighbor_stat(grid, method, w_size_xy=3)
    first = result["var"].isel(lead_time=0, y=slice(1, -1)).values
    assert np.isnan(first[:, :5]).all() and not np.isnan(first[:, 5:]).any()
    assert (first[:, 10:] == fill).all() and np.isfinite(first[:, 5:9]).all()
    expected = ens.neighbor_stat(grid.isel(lead_time=[1]), method, w_size_xy=3)
    inside = {"y": slice(1, -1), "x": slice(1, -1)}  # unstructured: other neighbors at the edges
    np.testing.assert_array_equal(result["var"].isel(lead_time=[1], **inside), expected["var"].isel(inside))


@pytest.mark.parametrize("w_size_xy,w_size_t", [(3, None), (4, 3), (None, 2)])
def test_neighbor_stat_regular_exact_mean(w_size_xy, w_size_t):
    """With NaN, mean of the valid values over the whole space x time window (truncated at the edges)"""

    grid = _grid_time()
    result = ens.neighbor_stat(grid, "mean", w_size_xy=w_size_xy, w_size_t=w_size_t)["var"].values
    values = grid["var"].values
    sizes = (w_size_t or 1, w_size_xy or 1, w_size_xy or 1)
    expected = np.full_like(values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        for index in np.ndindex(values.shape):  # centered as rolling: w // 2 before, (w - 1) // 2 after
            box = tuple(slice(max(i - w // 2, 0), i + (w - 1) // 2 + 1) for i, w in zip(index, sizes))
            expected[index] = np.nanmean(values[box])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)  # values of about 10
    assert np.isnan(result).any()


def test_neighbor_stat_unstructured_exact_mean_time():
    """Unstructured grid with a time window: the exact mean over space x time, as on the regular grid"""

    grid = _grid_lv95()
    expected = ens.neighbor_stat(grid, "mean", w_size_xy=3, w_size_t=2)
    flat = ens.neighbor_stat(_flat_shuffled(grid), "mean", w_size_xy=3, w_size_t=2)
    result = flat.set_index(cell=("y", "x")).unstack("cell").transpose(*expected["var"].dims).reindex_like(expected)
    inside = {"x": slice(1, -1), "y": slice(1, -1)}
    np.testing.assert_allclose(result["var"].isel(inside), expected["var"].isel(inside), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["mean", "max"])
def test_neighbor_stat_regular_dask(method):
    """Dask arrays stay lazy (chunks along the window dimensions merged), same result"""

    grid = _grid_time()
    expected = ens.neighbor_stat(grid, method, w_size_xy=3, w_size_t=3)
    lazy = ens.neighbor_stat(grid.chunk(lead_time=2, y=4, x=5), method, w_size_xy=3, w_size_t=3)
    assert lazy["var"].chunks is not None
    xr.testing.assert_identical(lazy.compute(), expected)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.int16, np.float32),
        (np.uint8, np.float32),
        (bool, np.float32),
        (np.int32, np.float64),
        (np.float64, np.float64),
    ],
)
@pytest.mark.parametrize("method", ["mean", "sum", "max"])
def test_neighbor_stat_dtype(dtype, expected, method):
    """float32 for float32, 8 or 16 bit integer and boolean data, float64 otherwise; values as the float data"""

    grid = _grid_time(nan=False)
    data = (grid > 0) if dtype is bool else grid.astype(dtype)
    result = ens.neighbor_stat(data, method, w_size_xy=3, w_size_t=2)
    assert result["var"].dtype == expected
    reference = ens.neighbor_stat(data.astype(np.float64), method, w_size_xy=3, w_size_t=2)
    xr.testing.assert_allclose(result.astype(np.float64), reference, rtol=1e-6)


def _grid_lv95(nan: bool = True) -> xr.Dataset:
    """Regular LV95 grid (1 km) with 2 lead times of random values, some NaN and an all-NaN 5 x 5 block"""

    rng = np.random.default_rng(0)
    values = rng.standard_normal((2, 12, 15)).astype(np.float32)
    if nan:
        values[rng.random(values.shape) < 0.2] = np.nan
        values[0, 3:8, 4:9] = np.nan
    return xr.Dataset(
        {"var": (("lead_time", "y", "x"), values)},
        coords={"x": 2_600_000.0 + 1000 * np.arange(15), "y": 1_200_000.0 + 1000 * np.arange(12)},
        attrs={"crs": "EPSG:2056"},
    )


def _flat_shuffled(grid: xr.Dataset) -> xr.Dataset:
    flat = grid.stack(cell=("y", "x")).reset_index("cell")
    return flat.isel(cell=np.random.default_rng(1).permutation(flat.sizes["cell"]))


@pytest.mark.parametrize("method", ["mean", "sum", "min", "max"])
@pytest.mark.parametrize("w_size", [3, 5])
def test_neighbor_stat_unstructured(method, w_size):
    """
    On an unstructured grid, the w_size**2 nearest points: the same as the rolling window of a square grid (inside,
    where the window is not truncated), including the NaN rule (NaN only if all the neighborhood is NaN)
    """

    half = w_size // 2
    inside = {"x": slice(half, -half), "y": slice(half, -half)}
    for nan in (False, True):
        grid = _grid_lv95(nan)
        expected = ens.neighbor_stat(grid, method, w_size_xy=w_size)
        if method == "mean" and nan:  # exact mean over the window (rolling along x then y: mean of the row means)
            windows = np.lib.stride_tricks.sliding_window_view(grid["var"].values, (w_size, w_size), axis=(1, 2))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
                expected = expected.copy(deep=True)
                expected["var"][:, half:-half, half:-half] = np.nanmean(windows, axis=(-2, -1))
        flat = ens.neighbor_stat(_flat_shuffled(grid), method, w_size_xy=w_size)
        assert flat["var"].dtype == np.float32
        result = flat.set_index(cell=("y", "x")).unstack("cell").transpose(*expected["var"].dims)
        result = result.reindex_like(expected)
        np.testing.assert_allclose(result["var"].isel(inside), expected["var"].isel(inside), rtol=1e-5, atol=1e-5)
        if nan and w_size == 3:  # the all-NaN 5 x 5 block gives NaN at its 3 x 3 center only
            assert result["var"].isel(lead_time=0, y=slice(4, 7), x=slice(5, 8)).isnull().all()
            assert result["var"].isel(lead_time=0).isnull().sum() == 9


@pytest.mark.parametrize("method", ["min", "max"])
def test_neighbor_stat_unstructured_blocks(method, monkeypatch):
    """min/max by blocks of points (180 points: blocks of 64, 64 and 52) equal one block"""

    flat = _flat_shuffled(_grid_lv95(nan=True))
    expected = ens.neighbor_stat(flat, method, w_size_xy=3)
    monkeypatch.setattr(ens, "BLOCK_BYTES", 1)
    xr.testing.assert_identical(ens.neighbor_stat(flat, method, w_size_xy=3), expected)


def test_neighbor_stat_unstructured_time_and_dask():
    """Unstructured grid with a time window, and dask arrays stay lazy"""

    flat = _flat_shuffled(_grid_lv95(nan=False))
    expected = ens.neighbor_stat(flat, "max", w_size_xy=3, w_size_t=2)
    lazy = ens.neighbor_stat(flat.chunk(lead_time=1), "max", w_size_xy=3, w_size_t=2)
    assert lazy["var"].chunks is not None
    xr.testing.assert_identical(lazy.compute(), expected)
    xr.testing.assert_identical(expected.coords.to_dataset(), flat.coords.to_dataset())


def test_upscaled_prob_unstructured_latlon():
    """upscaled_prob on an unstructured latitude/longitude grid (e.g. ICON), neighbors searched in meters"""

    grid = _grid_lv95(nan=False).expand_dims(realization=3)
    flat = _flat_shuffled(grid)
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True).transform(flat.x.values, flat.y.values)
    flat_latlon = flat.drop_vars(("x", "y")).assign_coords(longitude=("cell", lon), latitude=("cell", lat))
    flat_latlon.attrs = {}
    result = ens.upscaled_prob(flat_latlon, 0.5, "prob", w_size_xy=3)
    expected = ens.upscaled_prob(flat, 0.5, "prob", w_size_xy=3)
    np.testing.assert_array_equal(result["prob"].values, expected["prob"].values)
