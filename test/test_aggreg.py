# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import xarray as xr
from pandas import date_range, to_datetime
import grid_process.aggreg as agg


def test_reshape4intervals(model_data_validtime):
    """Simple testcase for the reshape4intervals function from aggreg module"""

    results = agg.reshape4intervals(model_data_validtime, 24, 24, 0)

    assert len(results.time) == np.sum(np.unique(model_data_validtime["time.day"], return_counts=True)[1] == 24)

    assert len(results.window) == 24

    expected = model_data_validtime.realization + xr.DataArray(np.append(np.arange(1, 24), 0), dims="window")
    assert (results.to_array() == expected).all()

    results = agg.reshape4intervals(model_data_validtime, 24, 12, 6)

    start = model_data_validtime.time[model_data_validtime["time.hour"] == 6][0].values
    stop = model_data_validtime.time[model_data_validtime["time.hour"] == 18][-1].values

    assert len(results.time) == np.ceil((stop - start) / np.timedelta64(1, "D"))
    assert np.all(results["time.hour"] == 18)
    assert len(results.window) == 12

    expected = model_data_validtime.realization + xr.DataArray(np.arange(7, 19), dims="window")
    assert (results.to_array() == expected).all()

    for interval in (3, 24):
        results = agg.reshape4intervals(model_data_validtime, 1, interval, 0)
        start = model_data_validtime.time[interval - 1].values
        stop = model_data_validtime.time[-1].values
        expected_t = xr.DataArray(date_range(start=start, end=stop, freq="1h"), dims="time")

        xr.testing.assert_equal(results.time, expected_t)
        assert len(results.window) == interval

    results = agg.reshape4intervals(model_data_validtime.isel(time=slice(0, 23)), 24, 24, 0)

    assert len(results.time) == 0
    assert len(results.window) == 1


def test_reshape4intervals_wtst(model_data_validtime):
    """Simple testcase for the reshape4intervals_wtst function from aggreg module"""

    results = agg.reshape4intervals_wtst(model_data_validtime, 24, 24, 0)

    start = model_data_validtime.time[model_data_validtime["time.hour"] == 23][0].values
    stop = model_data_validtime.time[model_data_validtime["time.hour"] == 22][-1].values
    n_st = np.ceil((stop - start) / np.timedelta64(1, "D"))

    start = model_data_validtime.time[model_data_validtime["time.hour"] == 0][0].values
    stop = model_data_validtime.time[model_data_validtime["time.hour"] == 23][-1].values
    n_wt = np.ceil((stop - start) / np.timedelta64(1, "D"))

    assert len(results.time) == n_st + n_wt
    assert model_data_validtime.source == results.source
    assert (results.time.diff("time")[::2] / np.timedelta64(1, "h") == 1).all()

    expected_st = model_data_validtime.realization + xr.DataArray(np.append(23, np.arange(23)), dims="window")
    expected_wt = model_data_validtime.realization + xr.DataArray(np.arange(24), dims="window")

    assert (results.where(results["time.hour"] == 22, drop=True).to_array() == expected_st).all()
    assert (results.where(results["time.hour"] == 23, drop=True).to_array() == expected_wt).all()

    results = agg.reshape4intervals_wtst(model_data_validtime, 1, 6, 0)
    expected = agg.reshape4intervals(model_data_validtime, 1, 6, 0)

    xr.testing.assert_equal(results, expected)


def test_split_by_timeres(poi_data_mixed_gran):
    """Simple testcase for the split_by_timeres function from aggreg module"""

    results = agg.split_by_timeres(poi_data_mixed_gran)
    assert len(results) == 2
    assert (results[0].time.diff("time") / np.timedelta64(10, "m") == 1).all()
    assert (results[1].time.diff("time") / np.timedelta64(1, "h") == 1).all()

    results = agg.split_by_timeres(poi_data_mixed_gran.isel(time=slice(0, 10)))
    assert len(results) == 1
    assert (results[0].time.diff("time") / np.timedelta64(10, "m") == 1).all()


def test_interval_stat(poi_data_mixed_gran):
    """Simple testcase for the interval_stat function from aggreg module"""

    agg_func = lambda x: x.max("window")
    results = agg.interval_stat(poi_data_mixed_gran, 24, 5, 3, agg_func, lt_duplicates=False)

    start = poi_data_mixed_gran.time[poi_data_mixed_gran["time.hour"] == 3][0].values
    stop = poi_data_mixed_gran.time[poi_data_mixed_gran["time.hour"] == 8][-1].values
    expected_t = xr.DataArray(
        date_range(start=start + np.timedelta64(5, "h"), end=stop, freq="1D"),
        dims="time",
    )

    xr.testing.assert_equal(results.time, expected_t)

    t_transit = to_datetime(
        poi_data_mixed_gran.time.where(poi_data_mixed_gran.time.diff("time") == np.timedelta64(10, "m"), drop=True)[
            -1
        ].values
    )

    expected_10min = (poi_data_mixed_gran["quantile"] + 57).astype(np.float32)
    expected_1hr = (poi_data_mixed_gran["quantile"] + 8).astype(np.float32)

    assert (results.sel(time=slice(None, t_transit)).to_array() == expected_10min).all()
    assert (results.sel(time=slice(t_transit, None)).to_array() == expected_1hr).all()

    agg_func = lambda x: x.sum("window")
    results = poi_data_mixed_gran.aggreg.interval_stat(6, 6, 0, agg_func, lt_duplicates=True)

    start = to_datetime(poi_data_mixed_gran.time[0].values).ceil("6h") + np.timedelta64(4, "h")
    stop = to_datetime(poi_data_mixed_gran.time[-1].values).floor("6h") - np.timedelta64(2, "h")
    expected_t_st = xr.DataArray(date_range(start=start, end=stop, freq="6h"), dims="time")

    start = to_datetime(poi_data_mixed_gran.time[0].values).ceil("6h") + np.timedelta64(5, "h")
    stop = to_datetime(poi_data_mixed_gran.time[-1].values).floor("6h") - np.timedelta64(1, "h")
    expected_t_wt = xr.DataArray(date_range(start=start, end=stop, freq="6h"), dims="time")
    expected_t = expected_t_st.combine_first(expected_t_wt)

    xr.testing.assert_equal(results.time, expected_t)

    time_exclude = poi_data_mixed_gran.time.sel(time=slice(t_transit.floor("1h"), t_transit))[1:]

    poi_data = poi_data_mixed_gran.where(~poi_data_mixed_gran.time.isin(time_exclude), drop=True)

    expected_st = poi_data.resample({"time": "6h"}, closed="right", label="right", offset="4h").sum(keep_attrs=True)
    expected_wt = poi_data.resample({"time": "6h"}, closed="right", label="right", offset="5h").sum(keep_attrs=True)
    expected = (
        xr.merge(
            [expected_st, expected_wt],
            combine_attrs="override",
            compat="no_conflicts",
            join="outer",
        )
        .sel(time=expected_t)
        .transpose(*list(poi_data.dims))
    )

    xr.testing.assert_allclose(results, expected)

    times_sub = poi_data_mixed_gran.time.sel(time=slice(t_transit, None))[1::5]

    poi_data = poi_data_mixed_gran.where(poi_data_mixed_gran.time.isin(times_sub), drop=True)

    agg_func = lambda x: x.min("window")
    results = agg.interval_stat(poi_data, 3, 3, 0, agg_func, lt_duplicates=False, interp_method="linear")

    interp_t = date_range(start=times_sub[0].values, end=times_sub[-1].values, freq="1h")
    expected = (
        poi_data.interp(time=interp_t, method="linear")
        .resample({"time": "3h"}, closed="right", label="right")
        .min(keep_attrs=True)
        .transpose(*list(poi_data.dims))
    )

    start = (to_datetime(poi_data.time[0].values) + np.timedelta64(2, "h")).ceil("3h")
    stop = to_datetime(poi_data.time[-1].values).floor("3h")
    expected_t = xr.DataArray(date_range(start=start, end=stop, freq="3h"), dims="time")

    xr.testing.assert_equal(results.time, expected_t)
    xr.testing.assert_allclose(results, expected.sel(time=expected_t))


def test_mode(model_data_validtime):
    """Simple testcase for the mode function from aggreg module"""

    reshaped_ds = agg.reshape4intervals(model_data_validtime.sel(realization=[0]), 24, 24, 0)

    # when ties but ordered_mode is None, should take the first sorted element
    results = agg.mode(reshaped_ds)
    assert np.allclose(results.to_array(), 0)

    # when ties and ordered_mode specified, its order is respected
    ordered_mode = {var: np.arange(24)[::-1].astype(np.float32) for var in reshaped_ds}
    results = agg.mode(reshaped_ds, ordered_mode)
    assert np.allclose(results.to_array(), 23)

    for var in reshaped_ds:
        reshaped_ds[var][dict(window=3)] = 5

    results = agg.mode(reshaped_ds, ordered_mode)
    assert np.allclose(results.to_array(), 5)


def test_extended_datamodel(model_data_validtime, poi_data_mixed_gran):
    """
    Simple testcase for the aggreg module functions run on extended data model
    """

    from grid_process.data_model import VALIDTIME

    agg_func = lambda x: x.mean("window")

    for validtime in VALIDTIME:
        ds_ext = model_data_validtime.rename(time=validtime)
        ds_ext_poi = poi_data_mixed_gran.rename(time=validtime)

        ds_out = agg.interval_stat(ds_ext, 6, 12, 0, agg_func)
        ds_out_poi = agg.interval_stat(ds_ext_poi, 24, 24, 0, agg_func, lt_duplicates=False)

        assert validtime in ds_out.coords
        assert validtime in ds_out.dims
        assert len(ds_out[validtime]) < len(model_data_validtime["time"])
        ds_out2 = agg.interval_stat(model_data_validtime, 6, 12, 0, agg_func)
        xr.testing.assert_identical(ds_out.rename({validtime: "time"}), ds_out2)

        assert validtime in ds_out_poi.coords
        assert validtime in ds_out_poi.dims
        assert len(ds_out_poi[validtime]) < len(poi_data_mixed_gran["time"])
        ds_out_poi2 = agg.interval_stat(poi_data_mixed_gran, 24, 24, 0, agg_func, lt_duplicates=False)
        xr.testing.assert_identical(ds_out_poi.rename({validtime: "time"}), ds_out_poi2)
