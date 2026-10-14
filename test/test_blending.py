# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the timeseries module"""

import numpy as np
from pandas import to_datetime
import grid_process.blending as bld


def test_combine_models(poi_data_units):
    """Simple testcase for the combine_models function from blending module"""

    reftime_iso = to_datetime(poi_data_units.forecast_reference_time[-1].values).isoformat()
    increm = 1000
    ds_2 = poi_data_units + increm
    ds_2["test_var_1"].attrs.update(poi_data_units.test_var_1.attrs)
    ds_2["test_var_2"].attrs.update(
        {
            "units": "celsius",
            "availability_time": {"2000-01-03T12:00:00": "2000-01-03T16:00:00"},
        }
    )
    ds_2 = ds_2.assign_coords({"time": ds_2["time"] + np.timedelta64(2, "D")})
    ds_2.attrs.update({"source": "COSMO-2E", "tags": ["operational", "1 hour", "2 km", "forecast"]})

    combined_ds = bld.combine_models([poi_data_units, ds_2])

    assert combined_ds.attrs == {"source": {"COSMO-1E": reftime_iso, "COSMO-2E": reftime_iso}}
    assert combined_ds.test_var_1.attrs == {"units": poi_data_units.test_var_1.units}
    assert combined_ds.test_var_2.attrs == {"units": poi_data_units.test_var_2.units}

    assert (combined_ds.sel(time=slice(None, poi_data_units.time[-1])).to_array().values < 500).all()
    assert (
        combined_ds.sel(time=slice(poi_data_units.time[-1] + np.timedelta64(10, "s"), None)).to_array().values > 500
    ).all()

    combined_ds = bld.combine_models([ds_2, poi_data_units])

    assert combined_ds.attrs == {"source": {"COSMO-2E": reftime_iso, "COSMO-1E": reftime_iso}}
    assert combined_ds.test_var_1.attrs == {"units": ds_2.test_var_1.units}
    assert combined_ds.test_var_2.attrs == {"units": ds_2.test_var_2.units}
    assert combined_ds.sizes == ds_2.sizes
    assert (combined_ds.to_array().values == ds_2.to_array().values).all()

    combined_ds = bld.combine_models([ds_2, poi_data_units], strict_start=False)
    assert (combined_ds.sel(time=slice(ds_2.time[0], ds_2.time[-1])).to_array().values == ds_2.to_array().values).all()
    assert (
        combined_ds.sel(time=slice(None, ds_2.time[0] - np.timedelta64(10, "s"))).to_array().values
        == poi_data_units.sel(time=slice(None, ds_2.time[0] - np.timedelta64(10, "s"))).to_array().values
    ).all()

    combined_ds = bld.combine_models([poi_data_units])
    assert poi_data_units.attrs.keys() == ds_2.attrs.keys()
    assert combined_ds.attrs == {"source": {"COSMO-1E": reftime_iso}}
    assert combined_ds.test_var_1.attrs == {"units": poi_data_units.test_var_1.units}
    assert combined_ds.test_var_2.attrs == {"units": poi_data_units.test_var_2.units}
    assert (combined_ds.to_array().values == poi_data_units.to_array().values).all()


def test_cleaver(poi_data_units):
    """Simple testcase for the cleaver function from blending module"""

    ind_switch = 30
    h = 6
    tail_val = 10
    head_val = 1000
    ds_tail = poi_data_units.isel(time=slice(None, ind_switch)).assign_attrs(
        {"source": "INCA", "tags": ["operational", "1 km", "nowcasting", "10 minutes"]}
    )
    ds_tail = ds_tail.where(ds_tail > np.inf, tail_val)
    ref_tail = to_datetime(ds_tail.forecast_reference_time[-1].values).isoformat()
    ds_head = poi_data_units.isel(time=slice(ind_switch - 5, None))
    ds_head = ds_head.where(ds_head > np.inf, head_val)
    ds_head["test_var_2"].attrs.update({"units": "degC"})
    ref_head = to_datetime(ds_head.forecast_reference_time[-1].values).isoformat()

    combined_ds = bld.cleaver(ds_tail, ds_head, h=h)
    assert combined_ds.attrs == {"source": {"INCA": ref_tail, "COSMO-1E": ref_head}}
    assert combined_ds.test_var_1.attrs == {"units": ds_tail.test_var_1.units}
    assert combined_ds.test_var_2.attrs == {"units": ds_tail.test_var_2.units}

    assert (combined_ds.isel(time=slice(None, ind_switch)).to_array().values == ds_tail.to_array().values).all()

    combined_ds_head = combined_ds.isel(time=slice(ind_switch + h, None))
    assert (
        combined_ds_head.to_array().values == ds_head.sel(time=slice(combined_ds_head.time[0], None)).to_array().values
    ).all()

    combined_ds_transi = combined_ds.isel(time=slice(ind_switch - 1, ind_switch + h))
    expected_transi = np.arange(h + 1) * (head_val - tail_val) / h + tail_val
    assert (combined_ds_transi.to_array().values == expected_transi).all()


def test_extended_datamodel(model_data_units):
    """
    Simple testcase for the blending module functions run on extended data model
    """

    from grid_process.data_model import REFTIME, LEADTIME, VALIDTIME

    ds1 = model_data_units.isel(forecast_reference_time=0, lead_time=slice(None, 3))
    ds2 = model_data_units.isel(forecast_reference_time=0, lead_time=slice(3, None))
    ds2 = ds2.assign_attrs({"source": "COSMO-2E"})
    ds3 = model_data_units.isel(forecast_reference_time=0, lead_time=slice(2, None))
    ds3 = ds3.assign_attrs({"source": "COSMO-2E"})

    for validtime in VALIDTIME:
        for leadtime in LEADTIME:
            for reftime in REFTIME:
                ds_ext1 = ds1.rename(time=validtime, lead_time=leadtime, forecast_reference_time=reftime)

                ds_out = bld.combine_models([ds_ext1, ds2])
                assert validtime in ds_out.coords
                assert validtime in ds_out.dims
                assert reftime not in ds_out.coords
                assert "COSMO-1E" in ds_out.source
                assert "COSMO-2E" in ds_out.source

                ds_out = bld.cleaver(ds_ext1, ds3)
                assert validtime in ds_out.coords
                assert validtime in ds_out.dims
                assert reftime not in ds_out.coords
                assert "COSMO-1E" in ds_out.source
                assert "COSMO-2E" in ds_out.source

                ds_out = bld.cleaver(ds_ext1, ds2)
                assert validtime in ds_out.coords
                assert validtime in ds_out.dims
                assert reftime not in ds_out.coords
                assert "COSMO-1E" in ds_out.source
                assert "COSMO-2E" in ds_out.source
