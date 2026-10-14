# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the helpers module"""

import pytest
import numpy as np
import xarray as xr
from pyproj import CRS

import grid_process.data_model as dm


def test_get_t_dim(model_data, model_data_validtime):
    """
    Simple testcase for the get_t_dim function
    """

    for leadtime in dm.LEADTIME:
        for validtime in dm.VALIDTIME:
            ds_ext = model_data.rename(lead_time=leadtime, time=validtime)
            assert dm.get_t_dim(ds_ext) == leadtime
            assert dm.get_t_dim(ds_ext.test_var_1) == leadtime
            ds_ext_validtime = model_data_validtime.rename(time=validtime)
            assert dm.get_t_dim(ds_ext_validtime) == validtime
            assert dm.get_t_dim(ds_ext_validtime.test_var_1) == validtime


def test_get_temporal_dims(model_data, model_data_validtime):
    """
    Simple testcase for the get_temporal_dims function
    """

    for reftime in dm.REFTIME:
        for leadtime in dm.LEADTIME:
            for validtime in dm.VALIDTIME:
                ds_ext = model_data.rename(forecast_reference_time=reftime, lead_time=leadtime, time=validtime)
                ds_ext_validtime = model_data_validtime.rename(forecast_reference_time=reftime, time=validtime)
                assert dm.get_temporal_dims(ds_ext) == (reftime, leadtime, None)
                assert dm.get_temporal_dims(ds_ext.test_var_1) == (
                    reftime,
                    leadtime,
                    None,
                )
                assert dm.get_temporal_dims(ds_ext_validtime) == (None, None, validtime)
                assert dm.get_temporal_dims(ds_ext_validtime.test_var_1) == (
                    None,
                    None,
                    validtime,
                )


def test_get_temporal_coords(model_data, model_data_validtime):
    """
    Simple testcase for the get_temporal_coords function
    """

    for reftime in dm.REFTIME:
        for leadtime in dm.LEADTIME:
            for validtime in dm.VALIDTIME:
                ds_ext = model_data.rename(forecast_reference_time=reftime, lead_time=leadtime, time=validtime)
                ds_ext_validtime = model_data_validtime.rename(forecast_reference_time=reftime, time=validtime)
                assert dm.get_temporal_coords(ds_ext) == (reftime, leadtime, validtime)
                assert dm.get_temporal_coords(ds_ext.test_var_1) == (
                    reftime,
                    leadtime,
                    validtime,
                )
                assert dm.get_temporal_coords(ds_ext_validtime) == (
                    reftime,
                    None,
                    validtime,
                )
                assert dm.get_temporal_coords(ds_ext_validtime.test_var_1) == (
                    reftime,
                    None,
                    validtime,
                )


def test_get_reftime(model_data):
    """
    Simple testcase for the get_reftime function
    """

    for reftime in dm.REFTIME:
        ds_ext = model_data.rename(forecast_reference_time=reftime)
        assert dm.get_reftime(ds_ext) == reftime
        assert dm.get_reftime(ds_ext.test_var_1) == reftime
        assert dm.get_reftime(ds_ext, "coord") == reftime
        assert dm.get_reftime(ds_ext.test_var_1, "coord") == reftime

    ds_ext = model_data.rename(forecast_reference_time="dummy")
    ds_ext["dummy"].attrs.update({"standard_name": "forecast_reference_time"})
    assert dm.get_reftime(ds_ext) == "dummy"
    assert dm.get_reftime(ds_ext.test_var_1) == "dummy"
    assert dm.get_reftime(ds_ext, "coord") == "dummy"
    assert dm.get_reftime(ds_ext.test_var_1, "coord") == "dummy"


def test_get_leadtime(model_data):
    """
    Simple testcase for the get_leadtime function
    """

    for leadtime in dm.LEADTIME:
        ds_ext = model_data.rename(lead_time=leadtime)
        assert dm.get_leadtime(ds_ext) == leadtime
        assert dm.get_leadtime(ds_ext.test_var_1) == leadtime
        assert dm.get_leadtime(ds_ext, "coord") == leadtime
        assert dm.get_leadtime(ds_ext.test_var_1, "coord") == leadtime

    ds_ext = model_data.rename(lead_time="dummy")
    ds_ext["dummy"].attrs.update({"standard_name": "forecast_period"})
    assert dm.get_leadtime(ds_ext) == "dummy"
    assert dm.get_leadtime(ds_ext.test_var_1) == "dummy"
    assert dm.get_leadtime(ds_ext, "coord") == "dummy"
    assert dm.get_leadtime(ds_ext.test_var_1, "coord") == "dummy"


def test_get_validtime(model_data_validtime, model_data):
    """
    Simple testcase for the get_validtime function
    """

    for validtime in dm.VALIDTIME:
        ds_ext_validtime = model_data_validtime.rename(time=validtime)
        assert dm.get_validtime(ds_ext_validtime) == validtime
        assert dm.get_validtime(ds_ext_validtime.test_var_1) == validtime
        assert dm.get_validtime(ds_ext_validtime, "coord") == validtime
        assert dm.get_validtime(ds_ext_validtime.test_var_1, "coord") == validtime
        ds_ext = model_data.rename(time=validtime)
        assert dm.get_validtime(ds_ext) is None
        assert dm.get_validtime(ds_ext.test_var_1) is None
        assert dm.get_validtime(ds_ext, "coord") == validtime
        assert dm.get_validtime(ds_ext.test_var_1, "coord") == validtime

    ds_ext_validtime = model_data_validtime.rename(time="dummy")
    ds_ext_validtime["dummy"].attrs.update({"standard_name": "time"})
    assert dm.get_validtime(ds_ext_validtime) == "dummy"
    assert dm.get_validtime(ds_ext_validtime.test_var_1) == "dummy"
    ds_ext = model_data.rename(time="dummy")
    ds_ext["dummy"].attrs.update({"standard_name": "time"})
    assert dm.get_validtime(ds_ext) is None
    assert dm.get_validtime(ds_ext.test_var_1) is None
    assert dm.get_validtime(ds_ext, "coord") == "dummy"
    assert dm.get_validtime(ds_ext.test_var_1, "coord") == "dummy"

    ds_ext_validtime["dummy"].attrs = {}
    ds_ext["dummy"].attrs = {}
    assert dm.get_validtime(ds_ext_validtime) is None
    assert dm.get_validtime(ds_ext_validtime.test_var_1) is None
    assert dm.get_validtime(ds_ext) is None
    assert dm.get_validtime(ds_ext.test_var_1) is None
    assert dm.get_validtime(ds_ext, "coord") is None
    assert dm.get_validtime(ds_ext.test_var_1, "coord") is None
    with pytest.raises(ValueError, match="Could not find a valid time dimension"):
        dm.get_validtime(ds_ext_validtime, raise_notfound=True)
        dm.get_validtime(ds_ext_validtime.test_var_1, raise_notfound=True)
        dm.get_validtime(ds_ext, raise_notfound=True)
        dm.get_validtime(ds_ext.test_var_1, raise_notfound=True)
        dm.get_validtime(ds_ext, "coord", raise_notfound=True)
        dm.get_validtime(ds_ext.test_var_1, "coord", raise_notfound=True)


def test_get_ensemble_dim(model_data):
    """
    Simple testcase for the get_ensemble_dim function
    """

    for ens_dim in dm.ENS_DIM:
        ds_ext = model_data.rename(realization=ens_dim)
        assert dm.get_ensemble_dim(ds_ext) == ens_dim
        assert dm.get_ensemble_dim(ds_ext.test_var_1) == ens_dim

    ds_ext = model_data.rename(realization="dummy")
    ds_ext["dummy"].attrs.update({"standard_name": "realization"})
    assert dm.get_ensemble_dim(ds_ext) == "dummy"
    assert dm.get_ensemble_dim(ds_ext.test_var_1) == "dummy"

    with pytest.raises(ValueError, match="Could not find an ensemble members dimension"):
        dm.get_ensemble_dim(model_data.rename(realization="dummy"))
        dm.get_ensemble_dim(model_data.test_var_1.rename(realization="dummy"))


@pytest.mark.parametrize(
    "data_cf_2dflat",
    ["2d_grid", "flat_grid"],
    indirect=True,
)
def test_set_grid_mapping(data_cf_2dflat):
    """Simple testcase for the set_grid_mapping function from helpers module"""

    results = dm.set_grid_mapping(data_cf_2dflat)
    if "points" in data_cf_2dflat.dims:
        assert {"lat", "lon"}.issubset(data_cf_2dflat.coords)
        assert {"lat", "lon"}.issubset(results.coords)
        assert "grid_mapping" not in results.attrs
    else:
        assert "grid_mapping" in results.attrs
        assert "grid_mapping" not in data_cf_2dflat.attrs
        assert isinstance(results.grid_mapping, str)
        assert "grid_mapping_dummy" in data_cf_2dflat

    assert "grid_mapping_dummy" not in results
    for varname, da in results.items():
        assert "grid_mapping" not in da.attrs
        assert "grid_mapping" in data_cf_2dflat[varname].attrs

    results_new = dm.set_grid_mapping(results)
    xr.testing.assert_identical(results_new, results)

    # Test error raised if more than one grid_mapping
    grid_mapping_1 = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
    grid_mapping_2 = xr.DataArray("a", attrs={"grid_mapping_name": "albers_conical_equal_area"})
    ds_multi_gridmap = data_cf_2dflat.assign({"grid_mapping_1": grid_mapping_1, "grid_mapping_2": grid_mapping_2})

    with pytest.raises(ValueError, match="Several grid mappings"):
        dm.set_grid_mapping(ds_multi_gridmap)


@pytest.mark.parametrize(
    "data_cf_2dflat",
    ["2d_grid", "flat_grid"],
    indirect=True,
)
def test_get_spatial_dims(data_cf_2dflat):
    """
    Simple testcase for the get_spatial_dims function
    """

    if "points" in data_cf_2dflat.dims:
        assert dm.get_spatial_dims(data_cf_2dflat) == ("points",)
        with pytest.raises(ValueError, match="Spatial dimensions must either be one of"):
            dm.get_spatial_dims(data_cf_2dflat.rename(points="dummy"))

    else:
        assert dm.get_spatial_dims(data_cf_2dflat) == ("y_1", "x_1")
        for y, x in dm.YX:
            ds_ext = data_cf_2dflat.rename(y_1=y, x_1=x)
            assert dm.get_spatial_dims(ds_ext) == (y, x)

        ds_ext = data_cf_2dflat.rename(y_1="bad_y", x_1="bad_x")
        assert dm.get_spatial_dims(ds_ext) == ("bad_y", "bad_x")
        ds_ext["bad_y"].attrs = {}
        ds_ext["bad_x"].attrs = {}
        with pytest.raises(ValueError, match="Spatial dimensions must either be one of"):
            dm.get_spatial_dims(ds_ext)


@pytest.mark.parametrize(
    "data_cf_2dflat",
    ["2d_grid", "flat_grid"],
    indirect=True,
)
def test_get_spatial_coords(data_cf_2dflat):
    """
    Simple testcase for the get_spatial_coords function
    """

    if "points" in data_cf_2dflat.dims:
        assert dm.get_spatial_coords(data_cf_2dflat) == ("lat", "lon")
        with pytest.raises(ValueError, match="Could not find non-dimension spatial coordinates"):
            dm.get_spatial_coords(data_cf_2dflat.rename(lat="dummy_lat"))

    else:
        assert dm.get_spatial_coords(data_cf_2dflat) == ("lat_1", "lon_1")
        for y, x in dm.YX:
            ds_ext = data_cf_2dflat.rename(lat_1=y, lon_1=x)
            assert dm.get_spatial_coords(ds_ext) == (y, x)

        ds_ext = data_cf_2dflat.rename(lat_1="bad_y", lon_1="bad_x")
        assert dm.get_spatial_coords(ds_ext) == ("bad_y", "bad_x")
        ds_ext["bad_y"].attrs = {}
        ds_ext["bad_x"].attrs = {}
        with pytest.raises(ValueError, match="Could not find non-dimension spatial coordinates"):
            dm.get_spatial_coords(ds_ext)


@pytest.mark.parametrize(
    "data_cf_2dflat",
    ["2d_grid", "flat_grid"],
    indirect=True,
)
def test_get_flat_dim(data_cf_2dflat):
    """
    Simple testcase for the get_flat_dim function
    """

    if "points" in data_cf_2dflat.dims:
        assert dm.get_flat_dim(data_cf_2dflat) == "points"

    else:
        assert dm.get_flat_dim(data_cf_2dflat) is None


def test_get_latlon_coords(model_data):
    """
    Simple testcase for the get_latlon_coords function
    """

    assert dm.get_latlon_coords(model_data) == ("latitude", "longitude")
    for lat, lon in dm.LATLON:
        ds_ext = model_data.rename(latitude=lat, longitude=lon)
        assert dm.get_latlon_coords(ds_ext) == (lat, lon)

    ds_ext = model_data.rename(latitude="dummy_y", longitude="dummy_x")
    assert dm.get_latlon_coords(ds_ext) is None
    ds_ext["dummy_y"].attrs.update({"standard_name": "latitude"})
    ds_ext["dummy_x"].attrs.update({"standard_name": "longitude"})
    assert dm.get_latlon_coords(ds_ext) == ("dummy_y", "dummy_x")


def test_get_latlon_dims(data_native_lv95):
    """
    Simple testcase for the get_latlon_dims function
    """

    ds = data_native_lv95.drop_vars(["latitude", "longitude"])
    assert dm.get_latlon_dims(ds) is None
    for lat, lon in dm.LATLON:
        ds_ext = ds.rename(y=lat, x=lon)
        assert dm.get_latlon_dims(ds_ext) == (lat, lon)

    ds_ext = ds.rename(y="dummy_lat", x="dummy_lon")
    assert dm.get_latlon_dims(ds_ext) is None
    ds_ext["dummy_lat"].attrs.update({"standard_name": "latitude"})
    ds_ext["dummy_lon"].attrs.update({"standard_name": "longitude"})
    assert dm.get_latlon_dims(ds_ext) == ("dummy_lat", "dummy_lon")


def test_get_surf_alt(model_data):
    """
    Simple testcase for the get_surf_alt function
    """

    for surf_alt in dm.SURF_ALT:
        assert dm.get_surf_alt(model_data.rename(surface_altitude=surf_alt)) == surf_alt

    with pytest.raises(ValueError, match="Missing one of"):
        dm.get_surf_alt(model_data.rename(surface_altitude="dummy"))


def test_get_poi_alt(poi_def):
    """
    Simple testcase for the get_poi_alt function
    """

    if "height_masl" in poi_def.columns:
        for name in dm.POI_ALT_COL:
            poi_ext = poi_def.rename(columns={"height_masl": name})
            poi_alt = dm.get_poi_alt(poi_ext)
            assert poi_alt.dtype == np.float32
            assert len(poi_def) == len(poi_alt)
    else:
        poi_alt = dm.get_poi_alt(poi_def)
        assert poi_alt.dtype == np.float32
        assert len(poi_def) == len(poi_alt)

    poi_broken = poi_def.drop(columns=["height_masl"], errors="ignore")
    poi_broken["geometry"] = poi_broken["geometry"].force_2d()
    with pytest.raises(ValueError, match="GeoDataFrame must have a"):
        dm.get_poi_alt(poi_broken)


@pytest.mark.parametrize(
    "data_native_lv95_2dflat,data_native_rotlatlon_2dflat,data_cf_2dflat",
    [("2d_grid", "2d_grid", "2d_grid"), ("flat_grid", "flat_grid", "flat_grid")],
    indirect=True,
)
def test_get_coords_name_crs(data_native_lv95_2dflat, data_native_rotlatlon_2dflat, data_cf_2dflat):
    """
    Simple testcase for the get_coords_name_crs function
    """

    grid_y_x_crs = dm.get_coords_name_crs(data_native_lv95_2dflat)
    assert grid_y_x_crs == ("latitude", "longitude", CRS.from_string("EPSG:4326"))

    da = data_native_lv95_2dflat[list(data_native_lv95_2dflat)[0]]
    assert grid_y_x_crs == dm.get_coords_name_crs(da)

    grid_y_x_crs = dm.get_coords_name_crs(data_native_lv95_2dflat.drop_vars(("latitude", "longitude")))
    assert grid_y_x_crs == ("y", "x", CRS.from_string("EPSG:2056"))

    spatial_dims = dm.get_spatial_dims(data_native_rotlatlon_2dflat)
    grid_y_x_crs = dm.get_coords_name_crs(data_native_rotlatlon_2dflat, spatial_dims)
    assert grid_y_x_crs == ("latitude", "longitude", CRS.from_string("EPSG:4326"))

    da = data_native_rotlatlon_2dflat[list(data_native_rotlatlon_2dflat)[0]]
    assert grid_y_x_crs == dm.get_coords_name_crs(da, spatial_dims)

    grid_y_x_crs = dm.get_coords_name_crs(
        data_native_rotlatlon_2dflat.drop_vars(("latitude", "longitude")), spatial_dims
    )
    assert grid_y_x_crs[:2] == ("y", "x")
    assert grid_y_x_crs[2].is_geographic

    grid_y_x_crs = dm.get_coords_name_crs(data_cf_2dflat)
    if "points" in data_cf_2dflat.dims:
        lat, lon = "lat", "lon"
    else:
        lat, lon = "lat_1", "lon_1"
    assert grid_y_x_crs == (lat, lon, CRS.from_string("EPSG:4326"))

    da = data_cf_2dflat[list(data_cf_2dflat)[0]]
    assert grid_y_x_crs == dm.get_coords_name_crs(da)

    if "cell" in data_native_lv95_2dflat.dims:
        pattern = "Could not find non-dimension spatial coordinates"
    else:
        pattern = "Missing spatial coordinates"
    with pytest.raises(ValueError, match=pattern):
        dm.get_coords_name_crs(
            data_native_lv95_2dflat.drop_vars(("latitude", "longitude", "y", "x", "cell"), errors="ignore")
        )


def test_config_extended():
    """
    Simple testcase to make sure that extending the data model by exporting
    ENV variables works.
    """

    assert "dummy_cell" in dm.FLAT_DIM
    assert ("dummy_y", "dummy_x") in dm.YX
    assert "dummy_surf_alt" in dm.SURF_ALT
    assert "dummy_height" in dm.POI_ALT_COL
    assert "dummy_time" in dm.VALIDTIME
    assert "dummy_leadtime" in dm.LEADTIME
    assert "dummy_reftime" in dm.REFTIME
    assert "dummy_ens_dim" in dm.ENS_DIM
