# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module for operations on timeseries data
"""

import logging
from functools import partial, reduce
from typing import Any, Literal, get_args
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.types import InterpOptions

from grid_process.helpers import make_accessor, keepattrs
from grid_process.data_model import (
    get_t_dim,
    get_temporal_coords,
    get_temporal_dims,
    get_latlon_coords,
    VALIDTIME,
)
from grid_process.ds_utils import cast_dtype
from grid_process.sun import daytime, zenith

logger = logging.getLogger(__name__)


# Register xarray accessor so module functions can also be called as methods
# on xr.Dataset objects
@xr.register_dataset_accessor("timeseries")
class Timeseries:
    """
    Xarray accessors class for operations on timeseries
    """

    def __init__(self, ds: xr.Dataset):
        self._data = ds


dataset_accessor = partial(make_accessor, accessor_class=Timeseries)


SunInterpOptions = Literal["sundur", "sunrad"]
ReindexOptions = Literal["nearest", "bfill", "ffill"]
InterpMethods = Literal["bcum", "fillval"] | SunInterpOptions | ReindexOptions | InterpOptions
FillnaMethods = Literal["bfill", "ffill", "fillval"] | InterpOptions
XR_INTERP_METHODS = reduce(lambda a, b: get_args(a) + get_args(b), get_args(InterpOptions))


@dataset_accessor
def interptogranularity(
    ds: xr.Dataset,
    granularity: str = "1h",
    method: InterpMethods = "linear",
    **kwargs: Any,
) -> xr.Dataset:
    """
    Interpolate the dataset to the specified time granularity. Where
    the dataset granularity is smaller than the specified one, the dataset
    granularity remains (no aggregation is performed).

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a temporal dimension expressed as numpy.datetime64 (valid time) or numpy.timedelta64 (lead time).
        Typical temporal dimensions names are supported, and can be extended via environment variables:

          - Valid time: export GRID_PROCESS__data_model__validtime (e.g., '["time", "valid_time", "dummy_time"]')
          - Lead time: export GRID_PROCESS__data_model__leadtime (e.g., '["lead_time", "step", "dummy_leadtime"]')
          - Forecast reference time: export GRID_PROCESS__data_model__reftime (e.g., '["forecast_reference_time",
            "ref_time", "reftime", "dummy_reftime"]')
          - When the temporal coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can
            be used directly.
    granularity : str, optional
        (maximum) granularity of the resulting time series. Must be one of the
        pandas.date_range ``freq`` options. Default is '1h'
    method : str, optional
        interpolation method for filling the time series. One of {'nearest', 'bfill',
        'ffill', 'bcum', 'fillval', 'sundur', 'sunrad', 'linear', 'zero', 'slinear',
        'quadratic', 'cubic', 'polynomial', 'barycentric', 'krogh', 'pchip', 'spline',
        'akima'}. Default is 'linear'
    **kwargs : any
        keyword arguments passed to the respective interpolation sub-function defined
        by the ``method``

    Returns
    -------
    ds_out : xarray.Dataset
        contains the full time series resulting from the time interpolation at
        the specified granularity

    See Also
    --------
    :func:`fill_na`, xarray.Dataset.reindex, xarray.Dataset.interp

    Note
    ----
    Consider using 'slinear' instead of 'linear' for similar results with possibly
    faster computations and lower memory footprint.
    """

    t_dim = get_t_dim(ds)
    if ds[t_dim].dtype.kind == "M":  # t is datetime64
        start_t = pd.to_datetime(min(ds[t_dim].values)).ceil(granularity)
        stop_t = pd.to_datetime(max(ds[t_dim].values))
        time_vec = pd.date_range(start_t, stop_t, freq=granularity)
    elif ds[t_dim].dtype.kind == "m":  # t is timedelta64
        start_t = pd.to_timedelta(min(ds[t_dim].values)).ceil(granularity)
        stop_t = pd.to_timedelta(max(ds[t_dim].values))
        time_vec = pd.timedelta_range(start_t, stop_t, freq=granularity)
    else:
        raise ValueError(f"Temporal dimension {t_dim} must be of numpy dtype datetime64 or timedelta64")

    if time_vec.isin(ds[t_dim].values).all():  # nothing to interpolate
        return ds

    time_vec = time_vec.union(ds[t_dim].values)
    t_coords = get_temporal_coords(ds)
    ref_t, lead_t, valid_t = t_coords

    if method == "bcum":
        ds_out = _bcum(ds, time_vec, t_dim, **kwargs)

    elif method == "fillval":
        ds_out = _fillval(ds, time_vec, (t_dim, ref_t, valid_t), **kwargs)

    elif method in get_args(SunInterpOptions):
        ds_out = _interp_sun(ds, time_vec, t_dim, method, **kwargs)  # type: ignore[arg-type]

    elif method in get_args(ReindexOptions):
        ds_out = ds.reindex({t_dim: time_vec}, method=method, **kwargs)  # type: ignore[arg-type]

    elif method in XR_INTERP_METHODS:
        # datetime-like variables along t_dim (e.g. reference time) are not interpolated: older xarray versions
        # drop them, recent ones interpolate them. Drop them explicitly, the reference time is reindexed below.
        datetime_vars = [
            name for name, var in ds.variables.items() if name != t_dim and t_dim in var.dims and var.dtype.kind in "Mm"
        ]
        src = ds.drop_vars(datetime_vars)
        ds_out = src.interp({t_dim: time_vec}, method, assume_sorted=True, kwargs=kwargs)  # type: ignore[arg-type]
        cast_dtype(ds_out, ds)

    else:
        raise ValueError(f"Temporal interpolation method {method!r} not supported")

    if (t_dim != valid_t) and (None not in t_coords):
        ds_out = ds_out.assign_coords({valid_t: ds[ref_t] + ds_out[lead_t]})
    elif (ref_t is not None) and (valid_t in ds[ref_t].dims) and (ref_t not in ds_out.coords):
        ds_out = ds_out.assign_coords({ref_t: ds[ref_t].reindex({valid_t: time_vec})})

    return ds_out


def _fillval(
    ds: xr.Dataset,
    time_vec: pd.TimedeltaIndex,
    t_dim_coords: tuple[str, str | None, str | None],
    fill_value: int | float | dict = 0.0,
    **reindex_kwargs: Any,
) -> xr.Dataset:
    """
    Temporal interpolation with specified fill value(s).
    Wrapper to xarray.Dataset.reindex with method=None.
    """

    dim, ref_t, valid_t = t_dim_coords
    if (valid_t is not None) and (dim != valid_t):
        to_drop = [valid_t]
    elif (ref_t is not None) and (valid_t in ds[ref_t].dims):
        to_drop = [ref_t]
    else:
        to_drop = []

    return ds.drop_vars(to_drop).reindex({dim: time_vec}, fill_value=fill_value, **reindex_kwargs)


@keepattrs
def _bcum(
    ds: xr.Dataset,
    time_vec: pd.TimedeltaIndex,
    dim: str,
    **reindex_kwargs: Any,
) -> xr.Dataset:
    """
    Temporal interpolation for accumulated quantities (such as precipitation amount).
    Values must represent the accumulated quantity since the last timestamp.
    """

    if ds.sizes[dim] <= 1:
        return ds

    model_t = ds[dim]
    time_vec_array = xr.DataArray(time_vec, coords={dim: time_vec})
    intervals = model_t.diff(dim) / time_vec_array.diff(dim)
    first = intervals[0].assign_coords({dim: model_t[0]})
    intervals = xr.concat((first, intervals), dim=dim).astype(np.float32)
    return (ds / intervals).reindex({dim: time_vec}, method="backfill", **reindex_kwargs)


def _interp_sun(  # pylint: disable=too-many-locals
    ds: xr.Dataset,
    time_vec: pd.TimedeltaIndex,
    dim: str,
    method: SunInterpOptions,
    lat: xr.DataArray | None = None,
    lon: xr.DataArray | None = None,
    **reindex_kwargs: Any,
) -> xr.Dataset:
    """
    Temporal interpolation for solar radiation and sunshine duration variables.
    Values must represent the accumulated quantity since the last timestamp.
    However, for solar radiation variables, it is assumed that values
    were transformed into fluxes beforehand (i.e. divided by the time delta). This
    is the convention used in MCH to have units of [W / m2] instead of [J / m2]
    to follow CF standards. This method disagregates accumulated values only to
    daytime timestamps and, for solar radiation, also rescale them according to
    the cosine of the solar hour angle. Accumulation is preserved.
    """

    if ds[dim].dtype.kind != "M":
        raise ValueError(
            f"Temporal interpolation with {method} method requires a temporal"
            " dimension with numpy.datetime64 (valid time) coordinates. Try to"
            " call the function setindex_validtime() first."
        )

    if ds.sizes[dim] <= 1:
        return ds

    if lat is None or lon is None:
        latlon = get_latlon_coords(ds, nondim=False)
        if latlon is not None:
            lat = ds[latlon[0]]
            lon = ds[latlon[1]]
        else:
            raise ValueError("Could not find latitude and longitude coordinates in the dataset")

    time_vec_array = xr.DataArray(time_vec, coords={dim: time_vec})
    first_interp = np.nonzero(~time_vec_array.isin(ds[dim]).values)[0][0]

    # cut ds in two parts. Only ds_end contains NaNs to be filled
    ds_start = ds.isel(time=slice(None, first_interp))
    ds_end = ds.isel(time=slice(first_interp, None))
    time_vec_end = time_vec_array.isel(time=slice(first_interp, None))

    # compute middle timestamp
    delta_t = time_vec_end.diff(dim).values
    delta_t = time_vec_end.values - np.concatenate((delta_t[[0]], delta_t)) / 2
    delta_t = xr.DataArray(delta_t, coords={dim: delta_t})

    # check daytime for middle timestamp and initialise correction factors
    if method == "sunrad":
        cor_fact = (
            np.cos(zenith(delta_t, lat, lon, refrac_cor=True).assign_coords({dim: time_vec_end}) * (np.pi / 180.0))
            .astype(np.float32)
            .clip(min=0.0)
        )
        isday = cor_fact > 0.0
        cor_stat = "mean"  # as radiation vars are transformed to flux in the internal grid database
    else:  # method is "sundur"
        isday = daytime(delta_t, lat, lon, refrac_cor=False).assign_coords(  # type: ignore[assignment]
            {dim: time_vec_end}
        )
        cor_fact = isday.astype(np.float32)
        cor_stat = "sum"

    # get model times for the end part
    model_t = [ds_start[dim][-1].values] + list(ds_end[dim].values)

    # divide factors by the mean of daytime timestamps between two model_t
    cor_list = []
    for i, t in enumerate(model_t[1:]):
        fact = cor_fact.isel(  # type: ignore[attr-defined]
            time=np.logical_and(time_vec_end > model_t[i], time_vec_end <= t)
        )
        fact /= getattr(fact, cor_stat)(dim)
        cor_list.append(fact)

    cor_fact = xr.concat(cor_list, dim=dim, coords="minimal", compat="override")

    # multiply radiation data by the scale factor
    ds_end = ds_end.reindex({dim: time_vec_end.values}, method="backfill", **reindex_kwargs) * cor_fact
    ds_end = ds_end.where(isday, 0.0)  # set night-time timestamps to zero

    return xr.concat((ds_start, ds_end), dim=dim, coords="minimal", compat="override")


@dataset_accessor
def fill_na(ds: xr.Dataset, method: FillnaMethods = "linear", **kwargs: Any) -> xr.Dataset:
    """
    Fill NaNs in the temporal dimension. Wrapper for different NaN
    interpolation routines.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a temporal dimension expressed as numpy.datetime64 (valid time) or numpy.timedelta64 (lead time).
        Typical temporal dimensions names are supported, and can be extended via environment variables:

          - Valid time: export GRID_PROCESS__data_model__validtime (e.g., '["time", "valid_time", "dummy_time"]')
          - Lead time: export GRID_PROCESS__data_model__leadtime (e.g., '["lead_time", "step", "dummy_leadtime"]')
          - When the temporal coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can
            be used directly.
    method : str
        interpolation method for filling the time series. One of {'bfill', 'ffill',
        'fillval', 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'polynomial', 'barycentric', 'krogh', 'pchip', 'spline', 'akima'}.
        Default is 'linear'
    **kwargs : any
        arguments passed to the respective NaN interpolation sub-function defined
        by the ``method``

    Returns
    -------
    ds_out : xarray.Dataset
        with NaNs filled. No extrapolation is performed, so NaNs on the side of
        the first and last real values remain NaNs

    See Also
    --------
    :func:`interptogranularity`, xarray.Dataset.interpolate_na, xarray.Dataset.bfill
    xarray.Dataset.ffill, xarray.Dataset.fillna
    """

    t_dim = get_t_dim(ds)

    custom_func = {
        "bfill": _bfill,  # type: ignore[dict-item]
        "ffill": _ffill,  # type: ignore[dict-item]
        "fillval": _fillna_val,
    }

    if method in custom_func:
        return custom_func[method](ds, t_dim, **kwargs)

    if method in XR_INTERP_METHODS:
        return ds.interpolate_na(t_dim, method, **kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Temporal NaN interpolation method {method!r} not supported")


def _bfill(ds: xr.Dataset, dim: str, **bfill_kwargs: Any) -> xr.Dataset:
    """
    Back fill NaNs in the temporal dimension. Wrapper to xarray.Dataset.bfill
    that prevents extrapolation at the begining of the timeseries.
    """

    def _set_nans(da_out: xr.DataArray) -> xr.DataArray:
        first_notnan = ds[da_out.name].isnull().argmin(dim)
        indexer = ds[dim].isel({dim: first_notnan})
        return da_out.where(da_out[dim] >= indexer)

    ds_out = ds.bfill(dim, **bfill_kwargs)
    return ds_out.map(_set_nans, keep_attrs=True)


def _ffill(ds: xr.Dataset, dim: str, **ffill_kwargs: Any) -> xr.Dataset:
    """
    Forward fill NaNs in the temporal dimension. Wrapper to xarray.Dataset.ffill
    that prevents extrapolation at the end of the timeseries.
    """

    def _set_nans(da_out: xr.DataArray) -> xr.DataArray:
        last_notnan = ds[da_out.name].isel({dim: slice(None, None, -1)}).isnull().argmin(dim)
        last_notnan = da_out.sizes[dim] - last_notnan - 1
        indexer = ds[dim].isel({dim: last_notnan})
        return da_out.where(da_out[dim] <= indexer)

    ds_out = ds.ffill(dim, **ffill_kwargs)
    return ds_out.map(_set_nans, keep_attrs=True)


def _fillna_val(ds: xr.Dataset, dim: str, fill_value: float | dict[str, float] = 0.0) -> xr.Dataset:
    """
    Fill NaNs in the temporal dimension with specified value(s).
    Wrapper to xarray.Dataset.fillna that prevents extrapolation at the begining
    and end of the timeseries.
    """

    def _set_nans(da_out: xr.DataArray) -> xr.DataArray:
        bool_da = ds[da_out.name].isnull()
        first_notnan = bool_da.argmin(dim)
        indexer_first = ds[dim].isel({dim: first_notnan})
        size_t = bool_da.sizes[dim]
        last_notnan = size_t - bool_da.isel({dim: np.arange(size_t)[::-1]}).argmin(dim) - 1
        indexer_last = ds[dim].isel({dim: last_notnan})
        return da_out.where((da_out[dim] >= indexer_first) & (da_out[dim] <= indexer_last))

    ds_out = ds.fillna(fill_value)
    return ds_out.map(_set_nans, keep_attrs=True)


@dataset_accessor
def setindex_validtime(ds: xr.Dataset, last_ref: bool = True) -> xr.Dataset:
    """
    Assign valid time dimension and coordinates (i.e. numpy.datetime64). The dataset
    can have zero, one or several forecast reference time and one or several
    lead time. In case several ref time and lead time are present, the function
    combines the different model runs by selecting from the most recent one where
    timestamps overlap.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a forecast reference time dimension and/or a lead time one
        Typical temporal dimensions names are supported, and can be extended via environment variables:

          - Lead time: export GRID_PROCESS__data_model__leadtime (e.g., '["lead_time", "step", "dummy_leadtime"]')
          - Forecast reference time: export GRID_PROCESS__data_model__reftime (e.g., '["forecast_reference_time",
            "ref_time", "reftime", "dummy_reftime"]')
          - When the temporal coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can
            be used directly.
    last_ref : bool
        whether to add the last forecast reference time. Used only if dimensions
        forecast reference time and lead time both exists and have length greater
        than 1. Default is True.

    Returns
    -------
    ds_out : xarray.Dataset
        with a valid time dimension labelled as numpy.datetime64.

    Note
    ----
    This function can be used to combine many runs of the same model into a
    continuous timeseries
    """

    def _slice_next_t(ds: xr.Dataset, ref: np.datetime64, next_t: np.datetime64) -> xr.Dataset:
        """
        Slice lead time up to the first timestamp of the next forecast reference time.
        """

        ds = ds.sel({ref_dim: ref})
        return ds.sel({lead_dim: (ds[valid_coord].values < next_t)})

    t_coords = get_temporal_coords(ds)
    ref_coord, lead_coord, valid_coord = t_coords
    ref_dim, lead_dim, valid_dim = get_temporal_dims(ds)

    if valid_dim is not None:
        return ds

    if sum(t is None for t in t_coords) > 1:
        raise ValueError(
            "The dataset needs at least two out of the 3 following temporal"
            " coordinates: valid time, ref time and lead time"
        )

    # single dim (lead_time or ref_time)
    if ref_dim is None:
        return _validtime_from_single_dim(ds, lead_dim, (valid_coord, ref_coord))  # type: ignore[arg-type]
    if lead_dim is None:
        return _validtime_from_single_dim(ds, ref_dim, (valid_coord, lead_coord), True, True)

    # single dim with size greater than one (lead_time or ref_time)
    if ds.sizes[ref_dim] == 1 and ds.sizes[lead_dim] >= 1:
        return ds.squeeze(ref_dim).timeseries.setindex_validtime()
    if ds.sizes[ref_dim] > 1 and ds.sizes[lead_dim] == 1:
        return ds.squeeze(lead_dim).timeseries.setindex_validtime()

    # both lead_time and ref_time dims have size greater than one
    if valid_coord is None:
        valid_coord = VALIDTIME[0]
        ds = ds.assign_coords({valid_coord: ds[ref_dim] + ds[lead_dim]})

    data = [
        _slice_next_t(ds, ref, next_t).timeseries.setindex_validtime()
        for ref, next_t in zip(ds[ref_dim].values, ds[valid_coord].isel({lead_dim: 0}).values[1:])
    ]

    if last_ref:
        data.append(ds.isel({ref_dim: -1}).timeseries.setindex_validtime())

    return xr.concat(data, dim=valid_coord, coords=[ref_dim], compat="override")


def _validtime_from_single_dim(
    ds: xr.Dataset,
    dim: str,
    coords: tuple[str | None, str | None],
    drop_coord: bool = False,
    dim_as_coord: bool = False,
) -> xr.Dataset:
    """
    Assign valid time dimension and coordinates for cases when only a lead time or reftime dimension is present.
    """

    valid_coord, other_coord = coords
    if valid_coord is None:
        times = (ds[dim] + ds[other_coord]).values
        valid_coord = VALIDTIME[0]
    else:
        times = ds[valid_coord].values
        ds = ds.drop_vars(valid_coord)

    ds_out = ds.rename({dim: valid_coord}).assign_coords({valid_coord: times})
    if dim_as_coord:
        ds_out = ds_out.assign_coords({dim: (valid_coord, ds[dim].values)})

    if other_coord is None:
        return ds_out

    if drop_coord:
        return ds_out.drop_vars(other_coord)

    return ds_out.assign_coords(
        {
            other_coord: (
                valid_coord,
                [ds[other_coord].values] * len(times),
            )
        }
    )


@dataset_accessor
def smoothing(ds: xr.Dataset, w: Sequence[float]) -> xr.Dataset:
    """
    Smooth timeseries by applying a moving weighted average.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a valid time or lead time dimension
    w : sequence of floats
        weights of the moving window

    Returns
    -------
    ds : xarray.Dataset
        contains the smoothed timeseries
    """

    dim = get_t_dim(ds)
    w_da = xr.DataArray(w / np.sum(w), dims=["window"]).astype(np.float32)
    w_len = len(w_da)
    return (
        ds.rolling({dim: w_len}, center=True).construct("window").map(lambda x: x.dot(w_da), keep_attrs=True).fillna(ds)
    )
