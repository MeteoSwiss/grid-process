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


def test_update_vars_attrs(model_data_units):
    """Simple testcase for the update_vars_attrs function from helpers module"""

    vars_attrs = {varname: da.attrs for varname, da in model_data_units.items()}

    for _, da in model_data_units.items():
        da.attrs = {}

    hlp.update_vars_attrs(model_data_units, vars_attrs)
    vars_attrs_2 = {varname: da.attrs for varname, da in model_data_units.data_vars.items()}
    assert vars_attrs == vars_attrs_2
