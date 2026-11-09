# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the helpers module"""

import numpy as np

import grid_process.helpers as hlp


def test_reproject():
    """Simple testcase for the reproject function from helpers module"""

    grid_coords_wgs84 = np.array(
        [
            [7.000000000, 46.600000000],
            [9.000000000, 47.000000000],
            [10.000000000, 46.700000000],
        ]
    )

    grid_coords_lv03 = np.array([[566390.413, 161065.044], [718743.875, 206619.409], [795862.269, 175284.624]])

    grid_coords_lv95 = np.array(
        [
            [2566390.024, 1161065.233],
            [2718744.552, 1206619.121],
            [2795863.175, 1175284.514],
        ]
    )

    x_lv95, y_lv95 = hlp.reproject(
        grid_coords_wgs84[:, 0],
        grid_coords_wgs84[:, 1],
        src_crs="EPSG:4326",
        dst_crs="EPSG:2056",
    )

    results_lv95 = np.c_[x_lv95, y_lv95]

    x_lv03, y_lv03 = hlp.reproject(
        grid_coords_lv95[:, 0],
        grid_coords_lv95[:, 1],
        src_crs="EPSG:2056",
        dst_crs="EPSG:21781",
    )

    results_lv03 = np.c_[x_lv03, y_lv03]

    x_wgs84, y_wgs84 = hlp.reproject(
        grid_coords_lv03[:, 0],
        grid_coords_lv03[:, 1],
        src_crs="EPSG:21781",
        dst_crs="EPSG:4326",
    )

    results_wgs84 = np.c_[x_wgs84, y_wgs84]

    assert np.allclose(grid_coords_lv95, results_lv95)
    assert np.allclose(grid_coords_lv03, results_lv03)
    assert np.allclose(grid_coords_wgs84, results_wgs84)


def test_sphere_as_wgs84():
    """
    Geographic CRS on a sphere are considered based on WGS84 (with the datum shift to other datums), other CRS
    are unchanged
    """

    from pyproj import CRS

    rotated = CRS.from_cf(
        {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 43.0,
            "grid_north_pole_longitude": -170.0,
            "earth_radius": 6371229.0,
        }
    )
    lonlat_sphere = CRS.from_cf({"grid_mapping_name": "latitude_longitude", "earth_radius": 6371229.0})
    for crs in ("EPSG:2056", CRS.from_epsg(21781), CRS.from_epsg(4326)):
        assert hlp.sphere_as_wgs84(crs) == CRS.from_user_input(crs)
    assert hlp.sphere_as_wgs84(lonlat_sphere) == CRS.from_epsg(4326)
    assert hlp.sphere_as_wgs84(rotated).to_json_dict()["base_crs"]["name"] == "WGS 84"
    # sphere given otherwise than by a CF grid mapping (e.g. "crs" attribute)
    assert hlp.sphere_as_wgs84("+proj=longlat +R=6371229 +no_defs") == CRS.from_epsg(4326)
    rotated_proj = "+proj=ob_tran +o_proj=longlat +o_lat_p=43 +o_lon_p=0 +lon_0=10 +R=6371229 +no_defs"
    assert hlp.sphere_as_wgs84(rotated_proj).to_json_dict()["base_crs"]["name"] == "WGS 84"

    # rotated pole to LV95 is the same as rotated pole to latitude/longitude, then WGS84 to LV95
    x_rot, y_rot = np.linspace(-2.0, 1.0, 50), np.linspace(-1.5, 1.5, 50)
    lon, lat = hlp.reproject(x_rot, y_rot, rotated, "EPSG:4326")
    expected = hlp.reproject(lon, lat, "EPSG:4326", "EPSG:2056")
    result = hlp.reproject(x_rot, y_rot, rotated, "EPSG:2056")
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-3)
    # and back
    np.testing.assert_allclose(hlp.reproject(*result, "EPSG:2056", rotated), (x_rot, y_rot), rtol=0, atol=1e-8)


def test_outer_ring():
    """
    Outer ring of a structured grid from 1D or 2D coordinates
    """

    x, y = np.arange(4.0), np.arange(3.0) * 10
    x_ring, y_ring = hlp.outer_ring(x, y)
    expected = {(xi, yi) for xi in x for yi in y if xi in (0, 3) or yi in (0, 20)}
    assert set(zip(x_ring, y_ring)) == expected and len(x_ring) == len(expected)
    x_2d, y_2d = np.meshgrid(x, y)
    np.testing.assert_array_equal(hlp.outer_ring(x_2d, y_2d), (x_ring, y_ring))


def test_update_vars_attrs(model_data_units):
    """Simple testcase for the update_vars_attrs function from helpers module"""

    vars_attrs = {varname: da.attrs for varname, da in model_data_units.items()}

    for _, da in model_data_units.items():
        da.attrs = {}

    hlp.update_vars_attrs(model_data_units, vars_attrs)
    vars_attrs_2 = {varname: da.attrs for varname, da in model_data_units.data_vars.items()}
    assert vars_attrs == vars_attrs_2
