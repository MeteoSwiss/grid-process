# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from pandas import date_range, timedelta_range
from pyproj import Transformer, CRS


LATLON_EPSG = "EPSG:4326"


def pytest_configure(config):
    """
    test that the config can be overwritten by env variables correctly and that
    extending the data model works throughout the code
    """

    os.environ["GRID_PROCESS__data_model__grid__unstructured_dim"] = '["cell", "points", "dummy_cell"]'
    os.environ["GRID_PROCESS__data_model__grid__y"] = '["y", "latitude", "lat", "dummy_y"]'
    os.environ["GRID_PROCESS__data_model__grid__x"] = '["x", "longitude", "lon", "dummy_x"]'
    os.environ["GRID_PROCESS__data_model__grid__surf_alt"] = (
        '["surface_altitude", "HSURF", "altitude", "dummy_surf_alt"]'
    )
    os.environ["GRID_PROCESS__data_model__poi__alt_from"] = (
        '["height_masl", "height", "altitude", "elevation", "dummy_height"]'
    )
    os.environ["GRID_PROCESS__data_model__validtime"] = '["time", "valid_time", "dummy_time"]'
    os.environ["GRID_PROCESS__data_model__leadtime"] = '["lead_time", "step", "dummy_leadtime"]'
    os.environ["GRID_PROCESS__data_model__reftime"] = (
        '["forecast_reference_time", "ref_time", "reftime", "dummy_reftime"]'
    )
    os.environ["GRID_PROCESS__data_model__ens_dim"] = '["realization", "eps", "epsd_1", "number", "dummy_ens_dim"]'


@pytest.fixture()
def model_data():
    """
    example of model data with dimensions `forecast_reference_time` and `lead_time`
    """

    x_dim = 15
    y_dim = 10
    time_dim = 8
    vertical_dim = 2
    real_dim = 11
    ref_time_dim = 3

    data_vals = (
        np.arange(0, y_dim * x_dim * ref_time_dim * time_dim * vertical_dim * real_dim)
        .reshape((y_dim, x_dim, ref_time_dim, time_dim, vertical_dim, real_dim))
        .astype(np.float32)
    )
    data_vars = {
        "test_var_1": (
            [
                "y",
                "x",
                "forecast_reference_time",
                "lead_time",
                "vertical",
                "realization",
            ],
            data_vals,
        ),
        "test_var_2": (
            [
                "y",
                "x",
                "forecast_reference_time",
                "lead_time",
                "vertical",
                "realization",
            ],
            data_vals * 2.5,
        ),
    }

    ref_time = date_range("2000-01-01", periods=ref_time_dim)
    leadtime = timedelta_range(0, periods=time_dim, freq="1h")
    time_matrix = [date_range(t, periods=time_dim, freq="1h") for t in ref_time]
    lon = np.tile(np.linspace(8, 9, x_dim), (y_dim, 1)).astype(np.float32)
    lat = np.tile(np.linspace(46.6, 47.3, y_dim), (x_dim, 1)).T.astype(np.float32)
    surface_altitude = np.random.rand(y_dim, x_dim) * 100 + 500

    coords = {
        "lead_time": ("lead_time", leadtime),
        "time": (["forecast_reference_time", "lead_time"], time_matrix),
        "longitude": (["y", "x"], lon),
        "latitude": (["y", "x"], lat),
        "surface_altitude": (["y", "x"], surface_altitude),
        "vertical": ("vertical", np.linspace(1000, 1020, vertical_dim)),
        "realization": ("realization", np.arange(real_dim)),
        "forecast_reference_time": ("forecast_reference_time", ref_time),
    }

    attrs = {
        "source": "COSMO-1E",
        "crs": LATLON_EPSG,
        "tags": ["operational", "1 hour", "2 km", "forecast"],
    }

    data = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    data["test_var_1"].attrs = {"units": "units_var_1"}
    data["test_var_2"].attrs = {"units": "units_var_2"}

    return data


@pytest.fixture()
def data_native_rotlatlon(model_data):
    """example of model data with native coordinates in rotated pole"""

    x = np.linspace(-1.5, -0.5, model_data.sizes["x"])
    y = np.linspace(-0.4, 0.3, model_data.sizes["y"])
    ds = model_data.assign_coords(x=x, y=y)
    ds.attrs["grid_mapping"] = (
        '{"earth_radius": 6371229.0, "grid_mapping_name": "rotated_latitude_longitude", "grid_north_pole_latitude": 43.0, "grid_north_pole_longitude": -170.0, "north_pole_grid_longitude": 0.0}'
    )
    del ds.attrs["crs"]

    x_2d = ds.x.broadcast_like(ds.longitude).values
    y_2d = ds.y.broadcast_like(ds.latitude).values
    transformer = Transformer.from_crs(CRS.from_cf(json.loads(ds.grid_mapping)), LATLON_EPSG, always_xy=True)
    ds["longitude"].values, ds["latitude"].values = transformer.transform(x_2d, y_2d)
    return ds


@pytest.fixture()
def data_native_lv95(model_data):
    """example of model data with native coordinates in swiss coordinates and constant resolution in x and y"""

    res = 2666
    x_start = 2675000
    x_stop = x_start + res * (model_data.sizes["x"] - 1)
    y_start = 1185000
    y_stop = y_start + res * (model_data.sizes["y"] - 1)
    x = np.linspace(x_start, x_stop, model_data.sizes["x"])
    y = np.linspace(y_start, y_stop, model_data.sizes["y"])
    ds = model_data.assign_coords(x=x, y=y)
    ds.attrs["grid_mapping"] = (
        '{"epsg_code": "EPSG:2056", "long_name": "Swiss coordinate system", "false_easting": 600000.0, "false_northing": 200000.0, "semi_major_axis": 6377397.155, "semi_minor_axis": 6356078.962822, "grid_mapping_name": "Oblique Mercator (CH1903)", "inverse_flattening": 299.1528128, "latitude_of_projection_center": 46.95240555555556, "longitude_of_projection_center": 7.439583333333333}'
    )
    del ds.attrs["crs"]

    x_2d = ds.x.broadcast_like(ds.longitude).values
    y_2d = ds.y.broadcast_like(ds.latitude).values
    transformer = Transformer.from_crs("EPSG:2056", LATLON_EPSG, always_xy=True)
    ds["longitude"].values, ds["latitude"].values = transformer.transform(x_2d, y_2d)
    return ds


@pytest.fixture()
def model_data_2dflat(request, model_data):
    if request.param == "flat_grid":
        return model_data.stack({"cell": ("y", "x")})
    return model_data


@pytest.fixture()
def data_native_lv95_2dflat(request, data_native_lv95):
    if request.param == "flat_grid":
        return data_native_lv95.stack({"cell": ("y", "x")})
    return data_native_lv95


@pytest.fixture()
def data_native_rotlatlon_2dflat(request, data_native_rotlatlon):
    if request.param == "flat_grid":
        return data_native_rotlatlon.stack({"cell": ("y", "x")})
    return data_native_rotlatlon


@pytest.fixture()
def model_data_units(model_data):
    """example of model data with units"""

    ds = model_data
    ds.test_var_1.attrs["units"] = "m s-1"
    ds.test_var_2.attrs["units"] = "°C"
    return ds


@pytest.fixture()
def data_cf_2dflat(request, data_native_rotlatlon):
    """example of model data with CF convention grid_mapping"""

    ds = data_native_rotlatlon.rename(
        {
            "x": "x_1",
            "y": "y_1",
            "longitude": "lon_1",
            "latitude": "lat_1",
            "surface_altitude": "HSURF",
        }
    )
    ds["x_1"].attrs.update({"axis": "X"})
    ds["y_1"].attrs.update({"axis": "Y"})
    ds["lon_1"].attrs.update({"standard_name": "longitude"})
    ds["lat_1"].attrs.update({"standard_name": "latitude"})
    ds.attrs.pop("source")
    ds.attrs.update(
        {
            "institution": "MeteoSwiss (NMC Switzerland)",
            "history": "Produced by someone",
        }
    )
    grid_mapping = json.loads(ds.attrs.pop("grid_mapping"))
    grid_mapping = {
        key: np.float32(value) if isinstance(value, float) else value for key, value in grid_mapping.items()
    }
    grid_mapname = "grid_mapping_dummy"
    for da in ds.values():
        da.attrs.update({"grid_mapping": grid_mapname})
    ds[grid_mapname] = xr.DataArray(0, attrs=grid_mapping)
    if request.param == "flat_grid":
        ds = ds.stack({"points": ("y_1", "x_1")})
        ds = ds.drop_vars([grid_mapname]).rename_vars({"lat_1": "lat", "lon_1": "lon"})
        ds["lon"].attrs = ds["lat"].attrs = ds["x_1"].attrs = ds["y_1"].attrs = {}
    elif request.param == "y_decreasing":
        ds = ds.isel({"y_1": slice(None, None, -1)})
    return ds


@pytest.fixture()
def model_data_validtime():
    """
    example of model data with dimension 'time' (numpy.datetime64)
    """

    x_dim = 15
    y_dim = 10
    real_dim = 11

    time = date_range(start="2000-01-01T22:00", end="2000-01-06T00:00", freq="1h")
    data_vals = np.tile(time.hour, (y_dim, x_dim, real_dim, 1)).astype(np.float32)
    data_vars = {
        "test_var_1": (["y", "x", "realization", "time"], data_vals),
        "test_var_2": (["y", "x", "realization", "time"], data_vals),
    }

    lon = np.tile(np.linspace(5, 11, x_dim), (y_dim, 1)).astype(np.float32)
    lat = np.tile(np.linspace(45, 49, y_dim), (x_dim, 1)).T.astype(np.float32)

    coords = {
        "time": ("time", time),
        "longitude": (["y", "x"], lon),
        "latitude": (["y", "x"], lat),
        "realization": ("realization", np.arange(real_dim)),
    }

    data = xr.Dataset(data_vars, coords=coords)
    data += data.realization.astype(np.float32)
    data["test_var_1"].attrs = {"units": "units_var_1"}
    data["test_var_2"].attrs = {"units": "units_var_2"}
    data = data.assign_coords({"forecast_reference_time": data.time})

    return data.assign_attrs({"source": "COSMO-1E", "crs": LATLON_EPSG})


@pytest.fixture(params=["alt_from_col", "alt_from_geom"])
def poi_def(request, model_data):
    """example of a pd.geodataframe with poi metadata"""

    n_poi = 100
    type_id = (np.ones((n_poi,)) * 3).astype(np.int64)
    point_id = np.arange(1, n_poi + 1)
    height_masl = np.random.randint(300, 2000, n_poi).astype(np.float64)

    q_lon = model_data.longitude.quantile(q=[0.2, 0.8], skipna=False).values
    q_lat = model_data.latitude.quantile(q=[0.2, 0.8], skipna=False).values
    lon = np.linspace(q_lon[0], q_lon[1], n_poi) + 0.001
    lat = np.linspace(q_lat[0], q_lat[1], n_poi) - 0.001

    gdf_poi = gpd.GeoDataFrame(
        {"type_id": type_id, "point_id": point_id, "height_masl": height_masl},
        geometry=gpd.points_from_xy(lon, lat),
    )

    gdf_poi["name"] = gdf_poi["type_id"].astype(str) + "_" + gdf_poi["point_id"].astype(str)

    gdf_poi.crs = LATLON_EPSG
    gdf_poi = gdf_poi[["type_id", "point_id", "name", "height_masl", "geometry"]]

    if request.param == "alt_from_geom":
        gdf_poi = gpd.GeoDataFrame(gdf_poi, geometry=gpd.points_from_xy(lon, lat, z=gdf_poi.height_masl)).drop(
            columns=["height_masl"]
        )

    return gdf_poi


@pytest.fixture()
def poi_def_rotlatlon(poi_def, data_native_rotlatlon):
    """example of a pd.geodataframe with poi metadata"""

    q_lon = data_native_rotlatlon.longitude.quantile(q=[0.2, 0.8], skipna=False).values
    q_lat = data_native_rotlatlon.latitude.quantile(q=[0.2, 0.8], skipna=False).values
    n_poi = len(poi_def)
    lon = np.linspace(q_lon[0], q_lon[1], n_poi) + 0.001
    lat = np.linspace(q_lat[0], q_lat[1], n_poi) - 0.001

    z = None
    if poi_def.iloc[0].geometry.has_z:
        z = poi_def.geometry.z.values

    gdf = gpd.GeoDataFrame(poi_def.drop(columns="geometry"), geometry=gpd.points_from_xy(lon, lat, z=z))
    gdf.crs = LATLON_EPSG
    return gdf


@pytest.fixture()
def poi_def_lv95(poi_def, data_native_lv95):
    """example of a pd.geodataframe with poi metadata"""

    q_lon = data_native_lv95.longitude.quantile(q=[0.2, 0.8], skipna=False).values
    q_lat = data_native_lv95.latitude.quantile(q=[0.2, 0.8], skipna=False).values
    n_poi = len(poi_def)
    lon = np.linspace(q_lon[0], q_lon[1], n_poi) + 0.001
    lat = np.linspace(q_lat[0], q_lat[1], n_poi) - 0.001

    z = None
    if poi_def.iloc[0].geometry.has_z:
        z = poi_def.geometry.z.values

    gdf = gpd.GeoDataFrame(poi_def.drop(columns="geometry"), geometry=gpd.points_from_xy(lon, lat, z=z))
    gdf.crs = LATLON_EPSG
    return gdf


@pytest.fixture()
def dummy_members(model_data):
    """
    example of model data with dimension 'realization' and constant values for each
    realization
    """

    data = model_data.copy()
    for var, da in data.items():
        data[var] = model_data.realization.broadcast_like(da).astype(np.float32)

    return data


@pytest.fixture()
def poi_data_mixed_gran():
    """
    example of poi data with dimension 'time' (numpy.datetime64) and a
    dimension 'quantile'. The temporal granularity is not constant
    """

    poi_dim = 15
    quantile_dim = 3

    time_10min = date_range(start="2000-01-01T18:20", end="2000-01-02T15:40", freq="10min")
    time_1hr = date_range(start=time_10min[-1].ceil("1h"), periods=48, freq="1h")

    time = time_10min.append(time_1hr)

    data_vals = np.tile(time.hour + time.minute, (poi_dim, quantile_dim, 1)).astype(np.float32)
    data_vars = {
        "test_var_1": (["poi", "quantile", "time"], data_vals),
        "test_var_2": (["poi", "quantile", "time"], data_vals),
    }

    coords = {
        "time": ("time", time),
        "poi": ("poi", np.arange(poi_dim)),
        "quantile": ("quantile", [0.1, 0.5, 0.9]),
    }

    data = xr.Dataset(data_vars, coords=coords)
    data += data["quantile"].astype(np.float32)
    data["test_var_1"].attrs = {"units": "units_var_1"}
    data["test_var_2"].attrs = {"units": "units_var_2"}
    data = data.assign_coords({"forecast_reference_time": data.time})

    return data.assign_attrs({"source": "COSMO-1E"})


@pytest.fixture()
def poi_data_units(poi_data_mixed_gran):
    """
    example of poi data with with units and more attributes
    """

    ds = poi_data_mixed_gran.where(poi_data_mixed_gran["time.minute"] == 0, drop=True)
    ds.test_var_1.attrs.update(
        {
            "units": "m s-1",
            "availability_time": {"2000-01-03T18:00:00": "2000-01-03T20:00:00"},
        }
    )
    ds.test_var_2.attrs.update(
        {
            "units": "°C",
            "availability_time": {"2000-01-03T18:00:00": "2000-01-03T20:00:00"},
        }
    )

    ds.attrs.update({"crs": LATLON_EPSG, "tags": ["operational", "1 hour", "1 km", "forecast"]})
    return ds


@pytest.fixture()
def horizon_data(model_data_validtime):
    """
    example of horizon data sampled regularly along 'azimuth' dim
    """

    lat = model_data_validtime.latitude
    x = np.arange(360, step=0.5)
    y = 12 * (1 + np.sin(2 * x * (np.pi / 180.0)))
    horizon = xr.DataArray(y, coords={"azimuth": x})
    horizon = horizon.broadcast_like(lat).copy()
    horizon[:2] = 1
    horizon[-2:] = 70

    return horizon
