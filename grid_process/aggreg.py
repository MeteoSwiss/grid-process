# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module to perform temporal aggregations
"""

import logging
from functools import partial
from typing import Any
from collections.abc import Callable, Iterable, Hashable

import numpy as np
import pandas as pd
import xarray as xr

from grid_process.helpers import make_accessor, keepattrs
from grid_process.data_model import get_validtime
from grid_process.ds_utils import cast_dtype
from grid_process.timeseries import interptogranularity, InterpMethods

logger = logging.getLogger(__name__)


# Register xarray accessor so module functions can also be called as methods
# on xr.Dataset objects
@xr.register_dataset_accessor("aggreg")
class Aggreg:
    """
    Xarray accessors class for aggregations
    """

    def __init__(self, ds: xr.Dataset):
        self._data = ds


dataset_accessor = partial(make_accessor, accessor_class=Aggreg)


@dataset_accessor
def reshape4intervals_wtst(ds: xr.Dataset, granularity: int, interval: int, offset: int = 0) -> xr.Dataset:
    """
    Wrapper for :func:`reshape4intervals` to reshape a dataset into intervals,
    additionally handling local time conversion and producing interval
    duplicates for summer and winter time.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with a valid time dimension to be reshaped. Must contain one constant
        time resolution
    granularity : int
        time in hours in between two consecutive interval starts
    interval : int
        time in hours of interval duration
    offset : int, optional
        time in hours by which interval starts should be shifted from 00:00 am.
        E.g. if 6, intervals include the one starting at 6am (4 and 5 am UTC)

    Returns
    -------
    ds_out : xarray.Dataset
        reshaped dataset with dimensions valid time x `window`. Valid time contains
        timestamps for respective interval, `window` has length of the interval
        and contains all timesteps belonging to the interval

    See also
    --------
    :func:`reshape4intervals`, :func:`interval_stat`
    """
    if granularity > 1:
        offset_utc_w = offset - 1
        ds_w = reshape4intervals(ds, granularity, interval, offset_utc_w)
        offset_utc_s = offset - 2
        ds_s = reshape4intervals(ds, granularity, interval, offset_utc_s)
        return xr.merge([ds_w, ds_s], combine_attrs="override", compat="no_conflicts", join="outer")

    return reshape4intervals(ds, granularity, interval, 0)


@dataset_accessor
def reshape4intervals(ds: xr.Dataset, granularity: int, interval: int, offset: int = 0) -> xr.Dataset:
    """
    Reshape a dataset into time intervals. The intervals are represented by
    the `window` dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with valid time dimension to be reshaped. Must contain one constant
        time resolution
    granularity : int
        time in hours in between two consecutive interval starts
    interval : int
        time in hours of interval duration
    offset : int, optional
        time in hours by which interval starts should be shifted from 00:00 am.
        E.g. if 6, intervals include the one starting at 6am

    Returns
    -------
    ds_out : xarray.Dataset
        reshaped dataset with dimensions valid time x `window`. Valid time
        contains timestamps for respective interval, `window` has length of the
        interval and contains all timesteps belonging to the interval

    See also
    --------
    :func:`reshape4intervals_wtst`, :func:`interval_stat`
    """

    t_dim = get_validtime(ds, raise_notfound=True)

    try:
        ds_ind = _get_valid_start(ds, granularity, offset, t_dim)
        interval_ind, granularity_ind = _adjust_resolution(ds_ind, interval, granularity, t_dim)

        indices = _get_intervalindices(ds_ind, interval_ind, granularity_ind, t_dim)
        return ds_ind.isel({t_dim: indices}).drop_vars(t_dim).rename({"time2": t_dim})

    except (IndexError, ValueError):
        logger.debug("Not enough timestamps to cover at least one aggregation interval")
        return ds.isel({t_dim: slice(None, 0)}).expand_dims("window")


def _get_valid_start(ds: xr.Dataset, granularity: int, offset: int, t_dim: str) -> xr.Dataset:
    """
    Get valid starting timestamp for reshaping a dataset into intervals, based
    on the granularity of the desired intervals, the offset and the initial
    time resolution of the data (does not require hourly resolution, just
    constant one throughout). Use to drop timestamps in the beginning belonging
    to 'incomplete' intervals.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset to be reshaped into intervals. Must contain one constant time
        resolution
    granularity : int
        time in hours in between two consecutive interval starts
    offset : int
        time in hours by which interval starts should be shifted from 00:00 am.
        E.g. if 6, intervals include the one starting at 6am
    t_dim : str
        name of the valid time dimension

    Returns
    -------
    ds_out : xarray.Dataset
        dataset cut at the first valid time from which one interval can be formed
    """

    time_res = ds[t_dim][1].values - ds[t_dim][0].values
    offset_td = np.timedelta64(offset, "h")
    start_t = pd.to_datetime(min(ds[t_dim].values)) - time_res - offset_td
    start_t = start_t.ceil(str(granularity) + "h") + time_res + offset_td

    return ds.sel({t_dim: slice(start_t, None)})


def _adjust_resolution(ds_ind: xr.Dataset, interval: int, granularity: int, t_dim: str) -> tuple[int, int]:
    """
    Adjust indexing lengths of granularity and interval before reshaping a
    dataset to required form for interval aggregations. This allow cases where
    time resolution of a dataset is not 1hr (e.g. if desired interval is 3hrs
    but time-resolution of the dataset is 10mins, index-length for aggregation
    should be 18 instead of 3).

    Parameters
    ----------
    ds_ind : xarray.Dataset
        dataset to be reshaped. Must contain one constant time resolution
    interval : int
        time in hours of interval duration
    granularity : int
        time in hours in between two consecutive interval starts
    t_dim : str
        name of the valid time dimension

    Returns
    -------
    interval_ind : int
        interval duration in amount of corresponding timesteps given the time
        resolution of ds_ind
    granularity_ind : int
        amount of corresponding timesteps in between two consecutive intervals,
        given the time resolution of ds_ind and the specified granularity
    """

    time_res = ds_ind[t_dim][1].values - ds_ind[t_dim][0].values
    interval_ind = np.timedelta64(interval, "h") / time_res
    granularity_ind = np.timedelta64(granularity, "h") / time_res

    if interval_ind % 1 != 0:
        raise ValueError("Interval must be a multiple of the dataset resolution")

    return int(interval_ind), int(granularity_ind)


def _get_intervalindices(ds_ind: xr.Dataset, interval_ind: int, granularity_ind: int, t_dim: str) -> xr.DataArray:
    """
    Get an array of indices referring to positions of timestamps to reshape
    a dataset according to desired interval and granularity.

    Parameters
    ----------
    ds_ind : xarray.Dataset
        dataset to be reshaped. Must contain one constant time-resolution and
        be cut to correct start time
    interval_ind : int
        interval duration in amount of corresponding timesteps given the time
        resolution of ds_ind
    granularity_ind : int
        amount of corresponding timesteps in between two consecutive intervals,
        given the time resolution of ds_ind and specified granularity
    t_dim : str
        name of the valid time dimension

    Returns
    -------
    da_ind : xarray.DataArray
        array of indices for reshaping with dims `time2` x `window` (newly formed
        intervals, labelled by their end-point) x (timesteps per interval)
    """

    indices = np.stack(
        [
            np.arange(ind, ind + interval_ind)
            for ind in np.arange(ds_ind.sizes[t_dim] - interval_ind + 1, step=granularity_ind)
        ]
    )

    coordinates = ds_ind[t_dim][indices[:, -1]].values
    return xr.DataArray(indices, dims=("time2", "window"), coords={"time2": coordinates})


def split_by_timeres(ds: xr.Dataset) -> list[xr.Dataset]:
    """
    Split a dataset where the time granularity changes. In case several
    switches of granularity are present, the split is done at the last
    granularity switch.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset with valid time dimension and non constant time resolution

    Returns
    -------
    out : list of two xarray.Dataset
        First element contains the first part of the dataset (with potentially
        another granularity switch hidden). Second element contains the last
        part of the dataset with constant granularity. In case the dataset is
        already at constant granularity or is of length 1 in valid time dimension,
        the output list has length 1
    """

    t_dim = get_validtime(ds, raise_notfound=True)

    if ds.sizes[t_dim] <= 1:
        return [ds]

    time_delta = ds[t_dim].diff(t_dim).values
    switch_gran = np.nonzero(time_delta != time_delta[-1])[0]

    if switch_gran.size > 0:
        switch = switch_gran[-1] + 1
        return [
            ds.isel({t_dim: slice(None, switch)}),
            ds.isel({t_dim: slice(switch, None)}),
        ]

    return [ds]


@dataset_accessor
@keepattrs
def interval_stat(
    ds: xr.Dataset,
    granularity: int,
    interval: int,
    offset: int,
    agg_func: Callable,
    lt_duplicates: bool = True,
    interp_method: InterpMethods | None = None,
    **agg_func_kwargs: Any,
) -> xr.Dataset:
    """
    Aggregate values in time over a moving window defined by a time
    ``granularity``, ``interval`` and ``offset``. Aggregation is based on the
    aggregating function ``agg_func``. The timeseries must have a granularity
    of maximum one hour or an ``interp_method`` must be specified for prior
    interpolation. The granularity does not need to be constant.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a valid time dimension, potentially with non-constant time resolution.
        Typical valid time dimension names are supported, and can be extended via environment variables:

          - Valid time: export GRID_PROCESS__data_model__validtime (e.g., '["time", "valid_time", "dummy_time"]')
          - When the coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can be
            used directly.
    granularity : int
        time in hours in between two consecutive interval starts
    interval : int
        time in hours of interval duration
    offset : int
        time in hours by which interval starts should be shifted from 00:00 am.
        E.g. if 6, intervals include the one starting after 6am
    agg_func : callable
        specify temporal aggregation function to use. The function must operate
        on `window` dimension. E.g. ``agg_func = lambda x: x.mean('window')``
    lt_duplicates : bool, optional
        if True, aggregations are done in the DWH compliant style, which means
        intervals are calculated with respect to both UTC+1 and UTC+2. Default
        is True
    interp_method : str, optional
        specify the temporal interpolation method to use if part of the
        timeseries has a greater time resolution than one hour. Default is no
        prior temporal interpolation (thus assuming the timeseries has max
        granularity of 1hr). Must be a method accepted by
        :func:`~grid_process.timeseries.interptogranularity`
    **agg_func_kwargs : any
        keyword arguments passed to the ``agg_func``

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with aggregated values. Uncomplete intervals are exluded.

    See also
    --------
    :func:`reshape4intervals_wtst`, :func:`reshape4intervals`,
    :func:`~grid_process.timeseries.interptogranularity`

    Note
    ----
    Alternatively, one can use xarray.Dataset.resample in some cases,
    especially when aggregating over uncomplete intervals is preferred.
    However, this behaviour can be problematic for cumulative aggregations
    (i.e. sum). Note that xarray.Dataset.resample has many more interesting
    options than this function although it does not allow to specify an
    interval and granularity separately.
    """

    if interp_method is not None:
        ds = interptogranularity(ds, "1h", interp_method)

    list_ds = split_by_timeres(ds)
    if len(list_ds) == 2:
        list_ds = [
            xr.merge(
                [interval_stat(ds, 1, 1, 0, agg_func, False, **agg_func_kwargs) for ds in list_ds],
                compat="no_conflicts",
                join="outer",
            )
        ]

    func = reshape4intervals_wtst if lt_duplicates else reshape4intervals
    ds_out = agg_func(func(list_ds[0], granularity, interval, offset), **agg_func_kwargs)

    cast_dtype(ds_out, ds)
    return ds_out


@dataset_accessor
def mode(reshaped_ds: xr.Dataset, ordered_mode: dict[Hashable, list[Any]] | None = None) -> xr.Dataset:
    """
    Take the most frequent category (mode) among categorical variables along
    `window` dimension. Could be used as input parameter ``agg_func`` for
    :func:`interval_stat`.

    Parameters
    ----------
    reshaped_ds : xarray.Dataset
        dataset with dimension `window` along which the mode is taken
    ordered_mode : dict {varname: list[any]}, optional
        ordered list of possible catergories for each data variable in the
        input dataset. If set, the returned category will match the priority
        order of this list in case of a tie between 2 or more categories

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with `window` dimension reduced

    See also
    --------
    :func:`interval_stat`, :func:`count`
    """

    ds = []
    for name, da in reshaped_ds.items():
        categ = None if ordered_mode is None else ordered_mode[name]
        counts = count(da, categ)
        cat_arr = np.array(list(counts))
        stacked_da = np.stack(list(counts.values()))

        ds.append(
            xr.DataArray(
                cat_arr[np.argmax(stacked_da, axis=0)],
                dims=[dim for dim in counts[cat_arr[0]].dims if dim != "window"],
            )
            .assign_coords(counts[cat_arr[0]].coords)
            .rename(name)
        )

    return xr.merge(ds).assign_attrs(reshaped_ds.attrs)


def count(reshaped_da: xr.DataArray, categ: Iterable | None = None) -> dict[Any, xr.DataArray]:
    """
    Count the number of occurences along the `window` dimension for each
    element present in the input DataArray or in the ``categ`` list.

    Parameters
    ----------
    reshaped_da : xarray.DataArray
        array with dimension `window` along which the count is done
    categ : Iterable, optional
        ordered iterable of categories for which to count the occurences. If not
        set, ``categ`` is the unique categories in the input DataArray

    Returns
    -------
    counts : dict of xarray.DataArray
        the keys are the different elements found in the array and the values
        are arrays with the `window` dimension reduced and containing the
        respective count

    See also
    --------
    :func:`mode`
    """

    counts = {}

    if categ is None:
        categ = np.unique(reshaped_da)

    for ct in categ:
        counts[ct] = (reshaped_da == ct).sum(dim="window")
    return counts
