# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module to wrap some time utility functions
"""

import logging
from functools import partial

from numpy import ndarray
from pandas import to_datetime, to_timedelta, DatetimeIndex
import xarray as xr

from grid_process.helpers import make_accessor
from grid_process.data_model import get_validtime

logger = logging.getLogger(__name__)


# Register xarray accessor so module functions can also be called as methods
# on xr.Dataset objects
@xr.register_dataset_accessor("time_utils")
class TimeUtils:
    """
    Xarray accessors class for time utility methods
    """

    def __init__(self, ds: xr.Dataset):
        self._data = ds


dataset_accessor = partial(make_accessor, accessor_class=TimeUtils)


def to_datetime_tzaware(timestamps: ndarray) -> DatetimeIndex:
    """
    Converts an array of numpy.datetime64 objects to a timezone-aware
    pandas.DatetimeIndex representing time in UTC.

    Parameters
    ----------
    timestamps : array of naive numpy.datetime64
        assumed to be in UTC time

    Returns
    --------
    out : pandas.DatetimeIndex
        UTC timezone aware
    """

    return to_datetime(timestamps).tz_localize("UTC")


@dataset_accessor
def set_daily(ds: xr.Dataset, lt_duplicates: bool = True, lt_hr: int = 0) -> xr.Dataset:
    """
    Sets timestamps of daily aggregations at the start of the day (i.e. 00:00).
    Daily parameters are stored in the DWH following this convention. If
    ``lt_duplicates=True``, selects correct validtimes between summer and
    winter local time duplicates first, otherwise assume daily aggregation are
    for UTC periods.

    Parameters
    ----------
    ds : xarray.Dataset
        with a valid time dimension (numpy.datetime64). When ``lt_duplicates=True``,
        duplicated timestamps corresponding to summer (UTC+2) and winter
        (UTC+1) localtime periods are expected as for a dataset returned by
        :func:`~grid_process.aggreg.interval_stat` with arg ``lt_duplicates``
        set to True
    lt_duplicates : bool
        whether the dataset contains duplicated timestamps corresponding
        to summer (UTC+2) and winter (UTC+1) localtime periods. Default is True
    lt_hr : int
        hour of the day representing the end of the daily period in localtime.
        Only used when ``lt_duplicates=True``. Default is 0, meaning the end
        of the period is 00:00 J+1 LT

    Returns
    -------
    ds_out : xarray.Dataset
        dataset where daily timestamps are set to 00:00 of the respective
        day. Optionally, selection of correct timestamps between summer/winter
        duplicates based on day of year is done

    See also
    --------
    :func:`~grid_process.aggreg.interval_stat`
    """

    validtime = get_validtime(ds, raise_notfound=True)
    if lt_duplicates:
        lt_dates = to_datetime_tzaware(ds[validtime].values).tz_convert("Europe/Zurich")

        ds_out = ds.isel({validtime: (lt_dates.hour == lt_hr)})
        return ds_out.assign_coords({validtime: ds_out[validtime].dt.floor("D")})

    return ds.assign_coords(
        {
            validtime: xr.where(
                ds[validtime].dt.hour == 0,
                ds[validtime] - to_timedelta("24h"),
                ds[validtime].dt.floor("D"),
            )
        }
    )
