# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module to compute ensemble statistics
"""

import logging
from typing import Literal, get_args
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import xarray as xr
from scipy import ndimage, sparse

from grid_process.helpers import make_accessor, keepattrs
from grid_process.data_model import get_t_dim, get_spatial_dims, get_ensemble_dim
from grid_process.aggreg import interval_stat
from grid_process.da_utils import da_to_ds
from grid_process.grid_interp import _flat_neighbors, _sparse_matmul

logger = logging.getLogger(__name__)

# bytes of results per block of points of _neighbor_extreme: small enough to stay in the CPU cache
BLOCK_BYTES = 2**18


# Register xarray accessor so module functions can also be called as methods
# on xr.Dataset objects
@xr.register_dataset_accessor("ens_stats")
class EnsStats:
    """
    Xarray accessors class for extracting probabilities from ensemble models
    """

    def __init__(self, ds: xr.Dataset):
        self._data = ds


dataset_accessor = partial(make_accessor, accessor_class=EnsStats)


NeighborStatMethods = Literal["mean", "sum", "min", "max"]


@dataset_accessor
def agg_prob(
    ds: xr.Dataset,
    dict_prob: dict,
    agg_func: Callable,
    lt_duplicates: bool = True,
    sign: str = "above",
) -> xr.Dataset:
    """
    Compute aggregated probabilities of exceedance. Temporal aggregation is
    done with :func:`~grid_process.aggreg.interval_stat`.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have valid time and ensemble dimensions and contain a single data variable.
        Typical dimensions names are supported, and can be extended via environment variables:

          - Valid time: export GRID_PROCESS__data_model__validtime (e.g., '["time", "valid_time", "dummy_time"]')
          - Ensemble: export GRID_PROCESS__data_model__ens_dim (e.g., '["realization", "eps", "epsd_1", "number",
            "dummy_ens_dim"]')
          - When the coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can be
            used directly.
    dict_prob : dict
        holds the keys 'prob_names', 'thresholds', 'granularity', 'interval',
        'offset' giving the specificities for the aggregated probabilities. If
        'prob_names' and 'thresholds' contain a list, the probabilites for
        several thresholds but same granularity, interval and offset are
        computed and stored in the same dataset as different variables.

            - dict_prob['prob_names']: str or list of str. Name(s) given to
              resulting data variable(s)
            - dict_prob['thresholds']: float or list of floats. Threshold to
              compute each probability
            - dict_prob['granularity']: int. Granularity in hours for the
              temporal aggregation
            - dict_prob['interval']: int. Interval in hours for the temporal
              aggregation
            - dict_prob['offset']: int. Offset in hours from 00:00 for the
              temporal aggregation

    agg_func : callable
        specify temporal aggregation function to use. The function must operate
        on `window` dimension. E.g. ``agg_func`` = lambda x: x.mean('window')
    lt_duplicates : bool, optional
        if True, aggregations are done in the DWH compliant style, which means
        intervals are calculated with respect to both UTC+1 and UTC+2. Default
        is True
    sign : {'above', 'below'}
        specify if we are looking for the probability to be above or below the
        thresholds. Default is 'above'

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with ensemble members dimension reduced and containing the
        aggregated probabilities of exceedance for each given threshold as data
        variables

    See also
    --------
    :func:`~grid_process.aggreg.interval_stat`, :func:`ens_prob`
    """

    agg_ds = interval_stat(
        ds,
        dict_prob["granularity"],
        dict_prob["interval"],
        dict_prob["offset"],
        agg_func,
        lt_duplicates,
    )

    return ens_prob(agg_ds, dict_prob["thresholds"], dict_prob["prob_names"], sign)


@dataset_accessor
def ens_prob(
    ds: xr.Dataset,
    thresholds: float | list[float],
    prob_names: str | list[str],
    sign: str = "above",
) -> xr.Dataset:
    """
    Compute probabilities of exceedance based on raw members.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have an ensemble dimension and contain a single data variable.
        Typical ensemble dimension names are supported, and can be extended via environment variables:

          - Ensemble: export GRID_PROCESS__data_model__ens_dim (e.g., '["realization", "eps", "epsd_1", "number",
            "dummy_ens_dim"]')
          - When the coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can be
            used directly.
    thresholds : float or list of floats
        the threshold(s) to compute the exceedence probabilities. Must be in
        the same units as the data.
    prob_names : str or list of str
        new variable name for each probability threshold
    sign : {'above', 'below'}
        specify if we are looking for the probability to be above or below the
        thresholds. Default is 'above'

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with the ensemble members dimension reduced and containing the
        probabilities of exceedance for each given threshold as a new data
        variable
    """

    if not isinstance(thresholds, list):
        thresholds = [thresholds]
    if not isinstance(prob_names, list):
        prob_names = [prob_names]

    thresh_da = xr.DataArray(thresholds, dims="variable", coords={"variable": prob_names})

    ens_dim = get_ensemble_dim(ds)
    if sign == "below":
        ds_out = (ds < thresh_da).mean(dim=ens_dim, skipna=False).astype(np.float32) * 100
    else:
        ds_out = (ds >= thresh_da).mean(dim=ens_dim, skipna=False).astype(np.float32) * 100

    vars_attrs = {varname: {"units": "percent"} for varname in prob_names}
    ds_out = da_to_ds(ds_out[list(ds.data_vars)[0]], vars_attrs)
    return ds_out.assign_attrs(ds.attrs)


@dataset_accessor
def neighbor_stat(
    ds: xr.Dataset,
    method: NeighborStatMethods,
    w_size_xy: int | None = None,
    w_size_t: int | None = None,
) -> xr.Dataset:
    """
    Compute a space x time neighborhood statistic.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have spatial and/or temporal dimensions.
        Typical dimensions names are supported, and can be extended via environment variables:

          - Valid time: export GRID_PROCESS__data_model__validtime (e.g., '["time", "valid_time", "dummy_time"]')
          - Lead time: export GRID_PROCESS__data_model__leadtime (e.g., '["lead_time", "step", "dummy_leadtime"]')
          - Y coordinate: export GRID_PROCESS__data_model__grid__y (e.g., '["y", "latitude", "lat", "dummy_y"]')
          - X coordinate: export GRID_PROCESS__data_model__grid__x (e.g., '["x", "longitude", "lon", "dummy_x"]')
          - Unstructured dimension: export GRID_PROCESS__data_model__grid__unstructured_dim (e.g., '["dummy_cell"]')
          - When the coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can be
            used directly.

        An unstructured grid needs spatial coordinates and a CRS, as for
        :func:`~grid_process.grid_interp.interp2grid`.
    method : {'mean', 'sum', 'min', 'max'}
        statistic applied over the moving window/volume
    w_size_xy : int, optional
        window size in the spatial dimensions. On an unstructured grid, the neighborhood of each point is its
        ``w_size_xy**2`` nearest points (itself included): on a square grid, the same as the window for 3 and 5.
    w_size_t : int, optional
        window size in the temporal dimension

    Returns
    -------
    ds_out : xarray.Dataset
        each value is the result of the neighborhood statistic

    Note
    ----
    The statistic is computed along each dimension of the window in turn (``scipy.ndimage`` filters along the
    dimensions of a regular grid and time, the neighbors along an unstructured dimension), exact for these methods
    but not for e.g. 'median' or 'std', which are not accepted.

    The windows are centered (as :meth:`xarray.Dataset.rolling` with ``center=True``) and truncated at the edges of
    regular grids and of the time dimension, whereas the neighborhoods of an unstructured grid always have
    ``w_size_xy**2`` points, extending further inside at its edges. NaN are ignored: a value is NaN only if all the
    values of its window are NaN, and 'mean' is the mean of the valid values over the whole space x time window.
    The result is float32 for float32, 8 or 16 bit integer and boolean data, float64 otherwise. On an unstructured
    grid, the nearest neighbors are searched in meters and cached (see :func:`~grid_process.grid_interp.interp2grid`).
    Dask arrays stay lazy, with a single chunk along the dimensions of the window.
    """

    ok_meth = get_args(NeighborStatMethods)
    if method not in ok_meth:
        raise ValueError(f"method must be one of {ok_meth} but was {method!r}")

    windows: dict[str, int | np.ndarray | sparse.csr_array] = {}  # window of each dimension, see _window_stat
    if w_size_xy is not None:
        spatial_dims = get_spatial_dims(ds)
        if len(spatial_dims) == 1:
            n_points = ds.sizes[spatial_dims[0]]
            k = min(w_size_xy**2, n_points)
            if k > 1:
                neighbors = _flat_neighbors(ds, spatial_dims[0], k)
                windows[spatial_dims[0]] = (
                    sparse.csr_array(
                        (np.ones(neighbors.size, np.float32), neighbors.ravel(), np.arange(0, neighbors.size + 1, k)),
                        shape=(n_points, n_points),
                    )
                    if method in ("mean", "sum")
                    else neighbors
                )
        else:
            windows.update(dict.fromkeys(spatial_dims, w_size_xy))
    if w_size_t is not None:
        windows[get_t_dim(ds)] = w_size_t

    def stat(da: xr.DataArray) -> xr.DataArray:
        dims = [dim for dim in windows if dim in da.dims]
        if not dims:
            return da
        out = xr.apply_ufunc(
            _window_stat,
            da,
            input_core_dims=[dims],
            output_core_dims=[dims],
            kwargs={"windows": [windows[dim] for dim in dims], "method": method},
            dask="parallelized",
            output_dtypes=[np.result_type(da.dtype, np.float32)],
            dask_gufunc_kwargs={"allow_rechunk": True},
            keep_attrs=True,
        )
        return out.transpose(*da.dims)

    return ds.map(stat, keep_attrs=True)


def _window_stat(
    values: np.ndarray, windows: list[int | np.ndarray | sparse.csr_array], method: NeighborStatMethods
) -> np.ndarray:
    """
    ``method`` over the window of each value along the last axes of ``values``, one per item of ``windows``: the
    window size along a regular axis (centered, truncated at the edges), or along an unstructured axis the 0/1 matrix
    of the neighbors of each point ('mean', 'sum') or their indices, shape (points, k) ('min', 'max'). NaN are
    ignored: NaN only if all the values of the window are NaN, 'mean' over the whole window. Computed and returned in
    ``np.result_type(values.dtype, np.float32)``.
    """

    dtype = np.result_type(values.dtype, np.float32)
    values = values.astype(dtype, copy=False)
    nan = np.isnan(values)
    has_nan = bool(nan.any())
    with np.errstate(invalid="ignore", divide="ignore"):
        if method in ("mean", "sum"):
            total = _separable_stat(np.where(nan, 0, values) if has_nan else values, windows, "sum")
            # valid values per window: of the window axes only without NaN
            valid = (~nan).astype(dtype) if has_nan else np.ones(values.shape[values.ndim - len(windows) :], dtype)
            counts = _separable_stat(valid, windows, "sum")
            if method == "mean":
                total /= counts
            else:
                np.copyto(total, np.nan, where=np.broadcast_to(counts == 0, total.shape))
            return total
        fill = -np.inf if method == "max" else np.inf
        out = _separable_stat(np.where(nan, fill, values) if has_nan else values, windows, method, fill)
        if has_nan:  # NaN where no valid value: where the fill value remains, unless there are genuine infinite values
            empty = (
                _separable_stat((~nan).view(np.uint8), windows, "max", 0) == 0
                if np.isinf(values).any()
                else out == fill
            )
            np.copyto(out, np.nan, where=empty)
        return out


def _separable_stat(
    values: np.ndarray, windows: list[int | np.ndarray | sparse.csr_array], method: str, fill: float = 0.0
) -> np.ndarray:
    """
    'sum', 'min' or 'max' of ``values`` (without NaN) along each of its last axes in turn, see :func:`_window_stat`.
    ``fill``: neutral value of 'min' or 'max' beyond the edges of the regular axes (+inf or -inf, 0 for a 0/1 mask).
    """

    for axis, window in zip(range(values.ndim - len(windows), values.ndim), windows):
        if isinstance(window, sparse.csr_array):  # sum over the neighbors of each point
            values = np.moveaxis(_sparse_matmul([window], np.moveaxis(values, axis, 0)), 0, axis)
        elif isinstance(window, np.ndarray):  # min or max over the neighbors of each point
            reduce_func = np.fmax if method == "max" else np.fmin
            values = np.moveaxis(_neighbor_extreme([window], np.moveaxis(values, axis, 0), reduce_func), 0, axis)
        elif method == "sum":
            values = ndimage.uniform_filter1d(values, window, axis, mode="constant", cval=0.0)
            values *= window
        else:
            filter_func = ndimage.maximum_filter1d if method == "max" else ndimage.minimum_filter1d
            values = filter_func(values, window, axis, mode="constant", cval=fill)
    return values


def _neighbor_extreme(neighbors: list[np.ndarray], values: np.ndarray, reduce_func: np.ufunc) -> np.ndarray:
    """
    ``reduce_func`` (:func:`numpy.fmin` or :func:`numpy.fmax`: NaN only if all are NaN) of the ``values`` (points
    first) of the neighbors of each point (rows of ``neighbors[0]``). By blocks of points, whose result stays in the
    CPU cache along the neighbor columns (3 to 5 times faster than whole columns).
    """

    index = neighbors[0]
    values = np.ascontiguousarray(values)  # each point a contiguous row (one copy unless stored points first)
    out = np.empty_like(values)
    block = max(64, BLOCK_BYTES // max(values[0].nbytes, 1))
    for start in range(0, len(index), block):
        part = out[start : start + block]
        rows = index[start : start + block]
        np.take(values, rows[:, 0], axis=0, out=part)
        for column in rows.T[1:]:
            reduce_func(part, values[column], out=part)
    return out


@dataset_accessor
def upscaled_prob(
    ds: xr.Dataset,
    thresholds: float | list[float],
    prob_names: str | list[str],
    w_size_xy: int | None = None,
    w_size_t: int | None = None,
) -> xr.Dataset:
    """
    Compute upscaled probabilities of exceedance. Upscaling is done in the
    space and/or time dimensions.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have an ensemble member dimension as well as a spatial and/or time one.
        Typical dimensions names are supported, and can be extended via environment variables:

          - Ensemble: export GRID_PROCESS__data_model__ens_dim (e.g., '["realization", "eps", "epsd_1", "number",
            "dummy_ens_dim"]')
          - Valid time: export GRID_PROCESS__data_model__validtime (e.g., '["time", "valid_time", "dummy_time"]')
          - Lead time: export GRID_PROCESS__data_model__leadtime (e.g., '["lead_time", "step", "dummy_leadtime"]')
          - Y coordinate: export GRID_PROCESS__data_model__grid__y (e.g., '["y", "latitude", "lat", "dummy_y"]')
          - X coordinate: export GRID_PROCESS__data_model__grid__x (e.g., '["x", "longitude", "lon", "dummy_x"]')
          - When the coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can be
            used directly.
    thresholds : float or list of floats
        the threshold(s) to compute the exceedence probabilities. Must be in
        the same units as the data
    prob_names : str or list of str
        new variable name for each probability threshold
    w_size_xy : int, optional
        window size for upscaling in the spatial dimensions
    w_size_t : int, optional
        window size for upscaling in the time dimension

    Returns
    -------
    ds_out : xarray.Dataset
        dataset with ensemble members dimension reduced and containing the
        upscaled probabilities of exceedance for each given threshold as a new
        data variable
    """

    ds = neighbor_stat(ds, "max", w_size_xy, w_size_t)
    return ens_prob(ds, thresholds, prob_names)


@dataset_accessor
def circular_mean(ds: xr.Dataset, dim: str | Sequence[str] | None = None) -> xr.Dataset:
    """
    Compute the mean of circular data between 0 and 360 degrees along the
    specified dimension(s).

    Parameters
    ----------
    ds : xarray.Dataset
    dim : str or sequence of str, optional
        dimension(s) on which to compute the circular mean. If None (default)
        look for the ensemble members dimension

    Returns
    -------
    ds_out : xarray.Dataset
    """

    if dim is None:
        dim = get_ensemble_dim(ds)

    sin_sum = np.sin(ds * np.pi / 180).sum(dim, skipna=False)  # type: ignore[call-overload]
    cos_sum = np.cos(ds * np.pi / 180).sum(dim, skipna=False)  # type: ignore[call-overload]
    res = np.arctan2(sin_sum, cos_sum) * 180 / np.pi + 360
    return res % 360


@dataset_accessor
@keepattrs
def circular_quantile(
    ds: xr.Dataset,
    q: float | Sequence[float] | np.ndarray,
    method: str = "linear",
    dim: str | None = None,
    wrap_around: bool = True,
) -> xr.Dataset:
    """
    Compute the qth quantile(s) of circular data (0 to 360 degrees) along the
    specified dimension.

    Parameters
    ----------
    ds : xarray.Dataset
    q : float in range of [0,1] or sequence of floats or ndarray of float
        quantile(s) to compute, which must be between 0 and 1 inclusive.
    method : str
        this parameter specifies the interpolation method to use when the desired
        quantile lies between two data points. See numpy.quantile for possible
        options. Default is 'linear'
    dim : str, optional
        dimension on which to compute the quantiles. If None (default) look for
        the ensemble members dimension
    wrap_around : bool
        whether to wrap around the results to the 0 to 360 interval. Default is
        True

    Returns
    -------
    ds_out : xarray.Dataset
        a `quantile` dimension replaces the dimension being reduced

    See Also
    --------
    numpy.quantile

    Note
    ----
    This approach was inspired by this answer on stackoverflow:
    https://stackoverflow.com/a/46453706. The simple intuition is that the
    opposite angle to the mean of the circular samples provides the reference
    point for their ranking.
    """

    if not isinstance(q, (Sequence, np.ndarray)):
        q = [q]

    if dim is None:
        dim = get_ensemble_dim(ds)

    ds_out = xr.Dataset()

    # compute circular mean
    ds_mean = circular_mean(ds, dim=dim)

    for varname, da in ds.items():
        # origin is the opposite point to the mean
        rank_origin = (ds_mean[varname] + 180) % 360
        # transform data
        transformed_da = da.where(da >= rank_origin, da + 360)
        # compute quantiles
        new_dims = list(da.dims)
        new_dims.remove(dim)
        new_dims = ["quantile"] + new_dims
        axis = da.get_axis_num(dim)
        tmp = np.quantile(transformed_da.values, q, axis=axis, method=method).astype(  # type: ignore[call-overload]
            da.dtype.type, copy=False
        )
        if wrap_around:
            tmp = tmp % 360
        ds_out[varname] = xr.DataArray(tmp, dims=new_dims, coords={"quantile": q})

    return ds_out.assign_coords(ds_mean.coords)
