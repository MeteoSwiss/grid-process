# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

import grid_process.units_conv as uco


def test_da_to(model_data_units):
    """Test unit conversion for data arrays"""

    da = model_data_units.test_var_1
    da_converted = da.units_conv.da_to("km / h")
    assert da_converted.attrs["units"]
    assert da_converted.units == "km / h"
    assert np.allclose(da * 3.6, da_converted)
    da = model_data_units.test_var_2
    da_converted = uco.da_to(da, "K")
    assert da_converted.attrs["units"]
    assert da_converted.units == "K"
    assert np.allclose(da, da_converted - 273.15)


def test_ds_to(model_data_units):
    """Test unit conversion for datasets"""

    ds_converted = model_data_units.units_conv.ds_to({"test_var_1": "km / h"})
    assert ds_converted.test_var_1.units == "km / h"
    assert np.allclose(model_data_units.test_var_1 * 3.6, ds_converted.test_var_1)
    assert model_data_units.test_var_2.units == "°C"
    assert (model_data_units.test_var_2 == ds_converted.test_var_2).all()
    ds_converted = model_data_units.units_conv.ds_to({"test_var_1": "km h-1", "test_var_2": "kelvin"})
    assert ds_converted.test_var_1.units == "km h-1"
    assert np.allclose(model_data_units.test_var_1 * 3.6, ds_converted.test_var_1)
    assert ds_converted.test_var_2.units == "kelvin"
    assert np.allclose(model_data_units.test_var_2, ds_converted.test_var_2 - 273.15)


def test_consistency_check(poi_data_units):
    """Test consistency_check function from units_conv module"""

    ds_2 = poi_data_units.copy()
    ds_2["test_var_1"].attrs.update(poi_data_units.test_var_1.attrs)
    ds_2["test_var_2"].attrs.update(
        {
            "units": "celsius",
            "availability_time": {"2000-01-03T12:00:00": "2000-01-03T16:00:00"},
        }
    )

    units_attrs = uco.consistency_check([poi_data_units, ds_2])
    assert units_attrs == {
        "test_var_1": {"units": "m s-1"},
        "test_var_2": {"units": "°C"},
    }
    units_attrs = uco.consistency_check([ds_2, poi_data_units])
    assert units_attrs == {
        "test_var_1": {"units": "m s-1"},
        "test_var_2": {"units": "celsius"},
    }

    with pytest.raises(ValueError):
        uco.consistency_check([poi_data_units, ds_2.drop_vars("test_var_2")])

    units_attrs = uco.consistency_check([poi_data_units, ds_2.drop_vars("test_var_2")], check_vars=False)
    assert units_attrs == {
        "test_var_1": {"units": "m s-1"},
        "test_var_2": {"units": "°C"},
    }

    ds_2["test_var_1"].attrs.update({"units": "km h-1"})

    with pytest.raises(uco.UnitsError):
        uco.consistency_check([poi_data_units, ds_2])


def test_units_check(model_data_units):
    """Test units_check function from units_conv module"""

    ds = model_data_units.copy()
    ds["test_var_2"].attrs.update({"units": "km h-1"})

    with pytest.raises(uco.UnitsError):
        uco.units_check([ds, ds])

    ds["test_var_1"].attrs.update({"units": "kilometer / hour"})
    uco.units_check([ds, ds])
    uco.units_check([ds, ds.drop_vars("test_var_2")])
