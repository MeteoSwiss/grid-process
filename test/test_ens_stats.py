# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ens_stats module"""

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
