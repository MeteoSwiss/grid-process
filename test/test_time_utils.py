# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the time_utils module"""

import numpy as np
import pandas as pd
import pytz
import datetime
import grid_process.time_utils as tmu


def test_to_datetime_tzaware(poi_data_mixed_gran):
    """
    Simple testcase for the to_datetime_tzaware function from time_utils
    module
    """

    results = tmu.to_datetime_tzaware(poi_data_mixed_gran.time.values)
    assert isinstance(results, pd.DatetimeIndex)
    assert (results.tzinfo == pytz.utc) or (results.tzinfo == datetime.timezone.utc)


def test_set_daily(poi_data_mixed_gran):
    """
    Simple testcase for the to_datetime_tzaware function from time_utils module
    """

    ds = poi_data_mixed_gran[dict(time=poi_data_mixed_gran.time.dt.minute == 0)]
    bool_22_23 = np.logical_or(ds.time.dt.hour == 22, ds.time.dt.hour == 23)
    bool_16_17 = np.logical_or(ds.time.dt.hour == 16, ds.time.dt.hour == 17)
    ds_duplicates_22_23 = ds.where(bool_22_23, drop=True)
    ds_duplicates_16_17 = ds.where(bool_16_17, drop=True)
    ds_00 = ds.where(ds.time.dt.hour == 0, drop=True)
    ds_18 = ds.where(ds.time.dt.hour == 18, drop=True)

    results_duplicates_22_23 = tmu.set_daily(ds_duplicates_22_23)
    results_duplicates_16_17 = tmu.set_daily(ds_duplicates_16_17, lt_hr=18)
    results_00 = tmu.set_daily(ds_00, lt_duplicates=False)
    results_18 = tmu.set_daily(ds_18, lt_duplicates=False)

    assert (results_duplicates_22_23.time.values == np.unique(ds_duplicates_22_23.time.dt.floor("D"))).all()
    assert (results_duplicates_22_23.time.dt.hour == 0).all()
    assert results_duplicates_22_23.sizes["time"] == int(sum(bool_22_23.values) / 2)

    assert (results_duplicates_16_17.time.values == np.unique(ds_duplicates_16_17.time.dt.floor("D"))).all()
    assert (results_duplicates_16_17.time.dt.hour == 0).all()
    assert results_duplicates_16_17.sizes["time"] == int(sum(bool_16_17.values) / 2)

    assert (results_00.time.values == ds_00.time - pd.to_timedelta("24h")).all()
    assert (results_00.time.dt.hour == 0).all()

    assert (results_18.time.values == ds_18.time.dt.floor("D")).all()
    assert (results_18.time.dt.hour == 0).all()


def test_extended_datamodel(model_data_validtime):
    """
    Simple testcase for the time_utils module functions run on extended data model
    """

    from grid_process.data_model import VALIDTIME

    assert "dummy_time" in VALIDTIME

    ds_day = model_data_validtime.isel(time=model_data_validtime.time.dt.hour == 0)
    ds_day_lt = model_data_validtime.isel(time=model_data_validtime.time.dt.hour.isin((22, 23)))

    for validtime in VALIDTIME:
        ds_ext = ds_day.rename(time=validtime)
        ds_ext_lt = ds_day_lt.rename(time=validtime)

        ds_out = tmu.set_daily(ds_ext, lt_duplicates=False)
        assert validtime in ds_out.coords
        assert validtime in ds_out.dims
        assert np.all(ds_out[validtime].diff(validtime).values == np.timedelta64(1, "D"))

        ds_out = tmu.set_daily(ds_ext_lt)
        assert validtime in ds_out.coords
        assert validtime in ds_out.dims
        assert np.all(ds_out[validtime].diff(validtime).values == np.timedelta64(1, "D"))
