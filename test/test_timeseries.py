# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the timeseries module"""

from datetime import datetime, timedelta, timezone

import pytest
import numpy as np
import pandas as pd
import xarray as xr

import grid_process.timeseries as tms


def test_interptogranularity(model_data, model_data_validtime):
    """
    Simple testcase for the interptogranularity function from timeseries module
    """

    # test with dim 'lead_time' labelled as timedelta

    # test linear interpolation
    new_data = np.broadcast_to(
        np.arange(model_data.sizes["lead_time"])[None, None, None, :, None, None],
        tuple(model_data.sizes.values()),
    ).astype(np.float32)

    expected = model_data.assign(
        {
            "test_var_1": (list(model_data.dims), new_data),
            "test_var_2": (list(model_data.dims), new_data),
        }
    )

    test_data = expected.isel(
        lead_time=np.append(
            np.arange(0, model_data.sizes["lead_time"], 2),
            model_data.sizes["lead_time"] - 1,
        )
    )

    assert tms.interptogranularity(test_data, "1h", "linear").equals(expected)

    # test bcum interpolation
    test_data = xr.full_like(test_data.isel(lead_time=slice(None, -1)), 2)

    results = tms.interptogranularity(test_data, "30min", "bcum")

    assert (len(test_data.lead_time) - 1) * 3 + len(test_data.lead_time) == len(results.lead_time)
    assert np.all(results.lead_time.diff("lead_time") == np.timedelta64(30, "m"))
    assert np.all(results.to_array() == 2 / 4)

    # verify time coords
    assert "time" in results.coords
    assert results.time.notnull().all()

    # test bfill interpolation
    test_data = xr.full_like(test_data, 1)

    results = test_data.timeseries.interptogranularity("1h", "bfill")

    assert len(results.lead_time) == (test_data.lead_time[-1] / np.timedelta64(1, "h") + 1)
    assert np.all(results.lead_time.diff("lead_time") == np.timedelta64(1, "h"))
    assert np.all(results.to_array() == 1)

    # test nearest interpolation
    results = tms.interptogranularity(test_data, "10min", "nearest")

    assert len(results.lead_time) == (test_data.lead_time[-1] / np.timedelta64(1, "h") * 6 + 1)
    assert np.all(results.lead_time.diff("lead_time") == np.timedelta64(10, "m"))
    assert np.all(results.to_array() == 1)

    # test with dim 'time' labelled as datetime

    # test linear interpolation
    new_data = np.broadcast_to(
        np.arange(model_data_validtime.sizes["time"])[:, None, None, None],
        tuple(model_data_validtime.sizes.values()),
    ).astype(np.float32)

    expected = model_data_validtime.assign(
        {
            "test_var_1": (list(model_data_validtime.dims), new_data),
            "test_var_2": (list(model_data_validtime.dims), new_data),
        }
    )

    test_data = expected.isel(time=np.arange(0, model_data_validtime.sizes["time"], 2))
    expected["forecast_reference_time"] = expected.forecast_reference_time.where(
        expected.forecast_reference_time.isin(test_data.forecast_reference_time)
    )

    xr.testing.assert_identical(tms.interptogranularity(test_data, "1h", "linear"), expected)

    # test bcum interpolation
    test_data = xr.full_like(test_data.isel(time=slice(None, -1)), 2)

    results = tms.interptogranularity(test_data, "30min", "bcum")

    assert (len(test_data.time) - 1) * 3 + len(test_data.time) == len(results.time)
    assert np.all(results.time.diff("time") == np.timedelta64(30, "m"))
    assert np.all(results.to_array() == 2 / 4)

    # test bfill interpolation
    test_data = xr.full_like(test_data, 1)

    results = tms.interptogranularity(test_data, "1h", "bfill")

    assert len(results.time) == len(test_data.time) * 2 - 1
    assert np.all(results.time.diff("time") == np.timedelta64(1, "h"))
    assert np.all(results.to_array() == 1)

    # test nearest interpolation
    results = tms.interptogranularity(test_data, "10min", "nearest")

    assert len(results.time) == (test_data.time[-1] - test_data.time[0]) / np.timedelta64(10, "m") + 1
    assert np.all(results.time.diff("time") == np.timedelta64(10, "m"))
    assert np.all(results.to_array() == 1)

    # test 'sundur' interpolation
    lat = test_data.latitude
    lon = test_data.longitude
    results = tms.interptogranularity(test_data, "30min", "sundur", lat=lat, lon=lon)

    assert len(results.time) == (test_data.time[-1] - test_data.time[0]) / np.timedelta64(30, "m") + 1
    assert np.all(results.time.diff("time") == np.timedelta64(30, "m"))
    results = results.isel(time=slice(1, None))
    aggreg = results.resample(time="2h", label="right", closed="right").sum()
    assert np.unique(aggreg.to_array())[1:] == 1
    assert np.all(
        results.where(
            results.time.dt.hour.isin([17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]),
            drop=True,
        ).to_array()
        == 0
    )
    assert np.allclose(results.where(results.time.dt.hour.isin([11, 12, 13]), drop=True).to_array(), 0.25)
    assert np.allclose(np.unique(results.to_array())[1:], 1 / np.array([4, 3, 2, 1]))

    # test 'sunrad' interpolation
    test_data = test_data.assign_coords(
        time=pd.date_range(start="2000-01-01T00:00", periods=len(test_data.time), freq="6h")
    )
    test_data.test_var_1[:] *= 800 * np.sin(np.pi / 24.0 * (test_data.time.dt.hour + 0.548))
    test_data.test_var_2[:] *= 300 * np.sin(np.pi / 24.0 * (test_data.time.dt.hour + 0.548))
    results = tms.interptogranularity(test_data, "1h", "sunrad", lat=lat, lon=lon)

    assert len(results.time) == (test_data.time[-1] - test_data.time[0]) / np.timedelta64(1, "h") + 1
    assert np.all(results.time.diff("time") == np.timedelta64(1, "h"))
    results = results.isel(time=slice(1, None))
    assert np.all(
        results.where(
            results.time.dt.hour.isin([17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]),
            drop=True,
        ).to_array()
        == 0
    )
    assert np.all(
        results.where(results.time.dt.hour == 11, drop=True).to_array().values
        > results.where(results.time.dt.hour == 10, drop=True).to_array().values
    )
    assert np.all(
        results.where(results.time.dt.hour == 15, drop=True).to_array().values
        > results.where(results.time.dt.hour == 16, drop=True).to_array().values
    )

    aggreg = results.resample(time="6h", label="right", closed="right").sum() * 1.0
    comp = test_data.isel(time=slice(1, None)).where(aggreg > 0, 0) * 6.0
    assert np.allclose(aggreg.to_array(), comp.to_array())  # accumulation preserved

    # test 'slinear' interpolation
    results = tms.interptogranularity(test_data, "3h", "slinear", copy=False)
    expected = tms.interptogranularity(test_data, "3h", "linear", copy=False)
    xr.testing.assert_allclose(results, expected)

    # test 'fillval' interpolation
    results = tms.interptogranularity(test_data, "30min", "fillval", fill_value=2)
    time_vec = xr.DataArray(
        np.full(results.time.shape, np.nan).astype(np.float32),
        coords=[("time", results.time.values)],
    )
    expected = tms.fill_na(test_data.combine_first(time_vec), "fillval", fill_value=2)
    xr.testing.assert_identical(results, expected)

    # special case: test data that is all nan
    test_data = xr.full_like(test_data, np.nan)
    results = tms.interptogranularity(test_data, "1h", "bcum")
    assert np.isnan(results.to_array()).all().item()
    assert len(results.time) == (test_data.time[-1] - test_data.time[0]) / np.timedelta64(1, "h") + 1

    # test error messages
    with pytest.raises(ValueError, match="Temporal interpolation method 'dummy' not supported"):
        tms.interptogranularity(model_data, "30min", "dummy")

    with pytest.raises(ValueError, match="must be of numpy dtype datetime64 or timedelta64"):
        ds_bad = model_data.assign_coords(lead_time=model_data.lead_time.astype(int))
        tms.interptogranularity(ds_bad, "1h", "dummy")


def test_fill_na(model_data_validtime):
    """
    Simple testcase for the interptogranularity function from timeseries module
    """

    test_data = model_data_validtime.isel(x=0, realization=0).copy(deep=True)
    test_data.test_var_1[{"time": [0, 1, 2, 3, -3, -2, -1], "y": slice(0, 5)}] = np.nan
    test_data[{"time": slice(10, 20)}] = np.nan

    expected = test_data.fillna(0)
    expected.test_var_1[{"time": [0, 1, 2, 3, -3, -2, -1], "y": slice(0, 5)}] = np.nan

    # test with "fillval" as the others methods are tested with interptogranularity()
    assert tms.fill_na(test_data, "fillval", fill_value=0).equals(expected)

    expected = test_data.bfill("time")
    expected.test_var_1[{"time": [0, 1, 2, 3, -3, -2, -1], "y": slice(0, 5)}] = np.nan
    assert tms.fill_na(test_data, "bfill").equals(expected)

    expected = test_data.ffill("time")
    expected.test_var_1[{"time": [0, 1, 2, 3, -3, -2, -1], "y": slice(0, 5)}] = np.nan
    assert tms.fill_na(test_data, "ffill").equals(expected)

    expected = test_data.interpolate_na("time", method="linear")
    assert tms.fill_na(test_data, "linear").equals(expected)


def test_setindex_validtime(model_data):
    """Simple testcase for the setindex_validtime function from timeseries module"""

    # test with several 'lead_time' and 'forecast_reference_time'
    results = tms.setindex_validtime(model_data)

    assert len(results.time) == len(model_data.lead_time) * len(model_data.forecast_reference_time)
    assert "time" in results.dims
    assert "lead_time" not in results.coords
    assert "forecast_reference_time" not in results.dims
    assert "time" in results.forecast_reference_time.dims
    assert np.all(np.unique(results.forecast_reference_time) == model_data.forecast_reference_time)
    assert np.issubdtype(results.time.dtype, np.datetime64)

    sliced = model_data.isel(lead_time=slice(1, None)).assign_coords(
        forecast_reference_time=pd.date_range(model_data.forecast_reference_time[0].values, freq="6h", periods=3)
    )

    sliced["time"] = sliced.forecast_reference_time + sliced.lead_time
    results = tms.setindex_validtime(sliced, last_ref=False)
    assert "time" in results.dims
    assert "lead_time" not in results.coords
    assert "forecast_reference_time" not in results.dims
    assert "time" in results.forecast_reference_time.dims
    assert np.all(results.time.diff("time") == np.timedelta64(1, "h"))
    assert sliced.forecast_reference_time.isin(results.time)[1:].all()
    assert results.time[-1] < sliced.time.isel(forecast_reference_time=-1, lead_time=0)

    # test with several 'lead_time' and one 'forecast_reference_time'
    results = tms.setindex_validtime(model_data.isel(forecast_reference_time=[0]))

    assert len(results.time) == len(model_data.lead_time)
    assert "time" in results.dims
    assert "lead_time" not in results.coords
    assert "forecast_reference_time" not in results.dims
    assert "forecast_reference_time" in results.coords
    assert "time" in results.forecast_reference_time.dims
    assert np.issubdtype(results.time.dtype, np.datetime64)
    assert np.all((results.time.values - results.forecast_reference_time.values) == model_data.lead_time)

    assert results.identical(tms.setindex_validtime(model_data.isel(forecast_reference_time=0)))
    # test with several 'forecast_reference_time' and one 'lead_time'
    results = tms.setindex_validtime(model_data.isel(lead_time=[0]))

    assert len(results.time) == len(model_data.forecast_reference_time)
    assert "time" in results.dims
    assert "lead_time" not in results.coords
    assert "forecast_reference_time" not in results.dims
    assert "forecast_reference_time" in results.coords
    assert "time" in results.forecast_reference_time.dims
    assert np.issubdtype(results.time.dtype, np.datetime64)
    assert np.all(results.time.values == model_data.forecast_reference_time.values)

    # test error messages
    with pytest.raises(
        ValueError,
        match="The dataset needs at least two out of the 3 following temporal",
    ):
        ds_bad = model_data.drop_vars(["time", "forecast_reference_time"])
        tms.setindex_validtime(ds_bad)


def test_smoothing():
    """Simple testcase for the smoothing function from timeseries module"""

    w = [0.5, 0, 0.5]
    t_beg = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    t_stop = t_beg + timedelta(hours=4 * 24 - 1)
    time_vec = pd.date_range(t_beg, t_stop, freq="1h")
    test_data = np.sin(np.arange(len(time_vec)) * 2 * np.pi / 24) * -1
    test_ds = (
        xr.DataArray(
            np.tile(test_data, (5, 3, 1)),
            coords=[
                ("poi", np.arange(5)),
                ("quantile", [0.1, 0.5, 0.9]),
                ("time", time_vec),
            ],
            name="test_data",
        )
        .to_dataset()
        .assign_attrs({"source": "COSMO-1E"})
    )
    test_ds["test_data"].attrs.update({"units": "units_test_data"})

    expected = [np.sum(test_data[ind - 1 : ind + 2] * w) for ind in np.arange(1, len(test_data) - 1)]
    expected.insert(0, test_data[0])
    expected.append(test_data[-1])
    expected = xr.DataArray(
        np.tile(expected, (5, 3, 1)),
        coords=[
            ("poi", np.arange(5)),
            ("quantile", [0.1, 0.5, 0.9]),
            ("time", time_vec),
        ],
        name="test_data",
    ).to_dataset()

    results = tms.smoothing(test_ds, w)
    assert np.all(results["test_data"].values == expected["test_data"].values)

    assert results.attrs == test_ds.attrs
    for var, da in results.items():
        assert da.attrs == test_ds[var].attrs


def test_extended_datamodel(model_data, model_data_validtime):
    """
    Simple testcase for the timeseries module functions run on extended data model
    """

    from grid_process.data_model import REFTIME, LEADTIME, VALIDTIME

    for reftime in REFTIME:
        for leadtime in LEADTIME:
            for validtime in VALIDTIME:
                ds_ext = model_data.rename(forecast_reference_time=reftime, lead_time=leadtime, time=validtime)
                ds_ext_validtime = model_data_validtime.rename(forecast_reference_time=reftime, time=validtime)

                ds_out = tms.interptogranularity(ds_ext, "30min", "linear")
                assert reftime in ds_out.dims
                assert leadtime in ds_out.dims
                assert validtime in ds_out.coords
                ds_out = tms.interptogranularity(ds_ext_validtime, "30min", "bfill")
                assert reftime in ds_out.coords
                assert validtime in ds_out.dims

                ds_out = tms.setindex_validtime(ds_ext)
                assert reftime in ds_out.coords
                assert validtime in ds_out.dims
                assert leadtime not in ds_out.coords
                ds_out = tms.setindex_validtime(ds_ext_validtime)
                assert reftime in ds_out.coords
                assert validtime in ds_out.dims

                ds_out = tms.smoothing(ds_ext, [0.2, 0.6, 0.2])
                assert leadtime in ds_out.dims
                ds_out = tms.smoothing(ds_ext_validtime, [0.2, 0.6, 0.2])
                assert validtime in ds_out.dims

                ds_out = tms.fill_na(ds_ext)
                assert reftime in ds_out.dims
                assert leadtime in ds_out.dims
                assert validtime in ds_out.coords
                ds_out = tms.fill_na(ds_ext_validtime)
                assert validtime in ds_out.dims
                assert reftime in ds_out.coords
