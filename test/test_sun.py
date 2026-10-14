# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the sun module"""

import numpy as np
from pandas import to_datetime
import xarray as xr
from astral import LocationInfo
from astral.sun import sunset, sunrise, zenith, azimuth, elevation
import grid_process.sun as sun
from grid_process.time_utils import to_datetime_tzaware


def test_daytime(poi_data_mixed_gran, poi_def, model_data_validtime):
    """
    Simple testcase for the daytime function from sun module
    """

    # check for POIs
    midnight_da = poi_data_mixed_gran.where(poi_data_mixed_gran.time.dt.hour == 0, drop=True).time
    midday_da = poi_data_mixed_gran.where(poi_data_mixed_gran.time.dt.hour == 12, drop=True).time
    lat = xr.DataArray(poi_def.geometry.y.values, dims=("poi",))
    lon = xr.DataArray(poi_def.geometry.x.values, dims=("poi",))
    assert not sun.daytime(midnight_da, lat, lon).any()
    assert sun.daytime(midday_da, lat, lon).all()

    times = poi_data_mixed_gran.time
    isday = sun.daytime(times, lat, lon, refrac_cor=False)
    isday2 = sun.elevation(times, lat, lon, refrac_cor=False) > 0
    assert isday.identical(isday2)

    # check for grid
    midnight_da = model_data_validtime.where(model_data_validtime.time.dt.hour == 0, drop=True).time
    midday_da = model_data_validtime.where(model_data_validtime.time.dt.hour == 12, drop=True).time

    lat = model_data_validtime.latitude
    lon = model_data_validtime.longitude
    assert not sun.daytime(midnight_da, lat, lon).any()
    assert sun.daytime(midday_da, lat, lon).all()

    times = model_data_validtime.time
    isday = sun.daytime(times, lat, lon, refrac_cor=False)
    isday2 = sun.elevation(times, lat, lon, refrac_cor=False) > 0
    assert isday.identical(isday2)


def test_zenith_elevation_azimuth(model_data_validtime):
    """
    Simple testcase for the zenith, cos_zenith, azim_elev and elevation functions from sun module
    """

    times = model_data_validtime.time
    lat = model_data_validtime.latitude
    lon = model_data_validtime.longitude
    times_tz = to_datetime_tzaware(times.values).to_pydatetime()
    obs = LocationInfo(latitude=lat[0, 0].values, longitude=lon[0, 0].values).observer

    # zenith
    expected_zen = xr.DataArray(
        np.array([zenith(obs, tt, with_refraction=True) for tt in times_tz]),
        coords=times.coords,
    )

    expected_zen.where(expected_zen < 90)  # could have larger differences at night due to methodology

    zen = sun.zenith(times, lat, lon, refrac_cor=True).isel(x=0, y=0)
    zen = zen.where(zen < 90)  # could have larger differences at night due to methodology
    assert np.abs(expected_zen - zen).max().values < 0.1

    expected_zen = xr.DataArray(
        np.array([zenith(obs, tt, with_refraction=False) for tt in times_tz]),
        coords=times.coords,
    )

    zen = sun.zenith(times, lat, lon, refrac_cor=False)
    assert np.abs(expected_zen - zen.isel(x=0, y=0)).max().values < 0.1

    # cos_zenith
    cos_zen = sun.cos_zenith(times, lat, lon)
    expected_cos_zen = np.cos(zen * (np.pi / 180.0))
    assert np.allclose(expected_cos_zen, cos_zen)

    # azim_elev
    expected_azim = xr.DataArray(np.array([azimuth(obs, tt) for tt in times_tz]), coords=times.coords)

    expected_elev = xr.DataArray(
        np.array([elevation(obs, tt, with_refraction=True) for tt in times_tz]),
        coords=times.coords,
    )

    expected_elev.where(expected_elev > 0)  # could have larger differences at night due to methodology

    azim, elev = sun.azim_elev(times, lat, lon, refrac_cor=True)
    azim = azim.isel(x=0, y=0)
    elev = elev.isel(x=0, y=0)
    assert np.abs(expected_azim - azim).max().values < 0.4
    assert np.abs(expected_elev - elev.where(elev > 0)).max().values < 0.1

    elev_2 = sun.elevation(times, lat, lon, refrac_cor=True).isel(x=0, y=0)
    assert elev.identical(elev_2)


def test_rise_set(poi_data_mixed_gran, poi_def):
    """
    Simple testcase for the rise_set function from sun module
    """

    times = poi_data_mixed_gran.time
    lat = xr.DataArray(poi_def.geometry.y.values, dims=("poi",))
    lon = xr.DataArray(poi_def.geometry.x.values, dims=("poi",))
    times_tz = to_datetime_tzaware(times.values).to_pydatetime()
    obs = LocationInfo(latitude=lat[-1].values, longitude=lon[-1].values).observer

    expected_sunrise = xr.DataArray(
        np.array([to_datetime(sunrise(obs, tt)).tz_localize(None) for tt in times_tz]),
        coords=times.coords,
    )

    expected_sunset = xr.DataArray(
        np.array([to_datetime(sunset(obs, tt)).tz_localize(None) for tt in times_tz]),
        coords=times.coords,
    )

    s_rise, s_set = sun.rise_set(times, lat, lon, refrac_cor=True)

    assert (np.abs(expected_sunrise - s_rise.isel(poi=-1)).max().values / np.timedelta64(1, "m")) < 3  # minutes

    assert (np.abs(expected_sunset - s_set.isel(poi=-1)).max().values / np.timedelta64(1, "m")) < 3  # minutes

    s_rise_nocor, s_set_nocor = sun.rise_set(times, lat, lon, refrac_cor=False)
    assert (s_rise < s_rise_nocor).all()
    assert np.abs(s_rise - s_rise_nocor).min() / np.timedelta64(1, "m") > 3  # minutes
    assert (s_set > s_set_nocor).all()
    assert np.abs(s_set - s_set_nocor).min() / np.timedelta64(1, "m") > 3  # minutes


def test_topo_rise_set(model_data_validtime, horizon_data):
    """
    Simple testcase for the topo_rise_set function from sun module
    """

    times = model_data_validtime.time[2:-1]  # remove partial first and last day
    lat = model_data_validtime.latitude
    lon = model_data_validtime.longitude

    ds_out = sun.topo_rise_set(times, lat, lon, horizon_data)

    assert (ds_out.sunset.isel(y=slice(2, -2)) > ds_out.sunrise.isel(y=slice(2, -2))).all()
    assert (ds_out.sunrise.isel(y=0) < ds_out.sunrise.isel(y=3)).all()
    assert ds_out.isel(y=[-2, -1]).isnull().to_array().all()
    assert "dayofyear" in ds_out.dims


def test_topo_shading_cor(model_data_validtime, horizon_data):
    """
    Simple testcase for the topo_shading_cor function from sun module
    """

    times = model_data_validtime.time
    lat = model_data_validtime.latitude
    lon = model_data_validtime.longitude

    da_out = sun.topo_shading_cor(times, lat, lon, horizon_data)

    assert (da_out.isel(y=slice(2, -2)) == 1).any("time").all()
    assert len(np.unique(da_out)) > 2
    assert (da_out.isel(y=[-2, -1]) == 0).all()


def test_extended_datamodel(model_data_validtime, horizon_data):
    """
    Simple testcase for the sun module functions run on extended data model
    """

    from grid_process.data_model import VALIDTIME

    lat = model_data_validtime.latitude
    lon = model_data_validtime.longitude

    for validtime in VALIDTIME:
        ds_ext = model_data_validtime.rename(time=validtime)
        times = ds_ext[validtime]

        ds_out = sun.topo_rise_set(times[2:-1], lat, lon, horizon_data)
        assert "dayofyear" in ds_out.coords
        assert "dayofyear" in ds_out.dims
        assert {"sunrise", "sunset"} == set(ds_out)

        ds_out = sun.topo_shading_cor(times, lat, lon, horizon_data)
        assert validtime in ds_out.coords
        assert validtime in ds_out.dims
        xr.testing.assert_identical(times.drop_vars("forecast_reference_time"), ds_out[validtime])
