# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module for blending datasets together
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from grid_process.helpers import replace_vars_attrs
from grid_process.data_model import get_t_dim, get_validtime, get_reftime
from grid_process.units_conv import consistency_check
from grid_process.timeseries import (
    setindex_validtime,
    interptogranularity,
    InterpMethods,
)

logger = logging.getLogger(__name__)


def combine_models(datasets: Sequence[xr.Dataset], strict_start: bool = True) -> xr.Dataset:
    """
    Combine datasets in valid time dimension. The new coordinates follow the normal
    broadcasting and alignment rules of ``join='outer'``. Vacant cells in the
    expanded coordinates are filled with numpy.nan .

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Datasets with valid time or lead time dimension to be combined. The sequence
        must reflect priority order, so where several datasets overlap in time, the
        data of the first one is kept
        Typical temporal dimensions names are supported, and can be extended via environment variables:

          - Valid time: export GRID_PROCESS__data_model__validtime (e.g., '["time", "valid_time", "dummy_time"]')
          - Lead time: export GRID_PROCESS__data_model__leadtime (e.g., '["lead_time", "step", "dummy_leadtime"]')
          - When the temporal coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can
            be used directly.
    strict_start : bool, optional
        If True (default), the resulting timeseries strictly starts with the
        dataset having highest priority (i.e. no older timestamps from lower
        priority datasets can be prepended)

    Returns
    -------
    combined_ds : xarray.Dataset
        contains the full timeseries resulting from the datasets combination.
        Valid time dimension is labelled by numpy.datetime64 although
        inputs could have lead time dimension (numpy.timedelta64)

    See also
    --------
    xarray.Dataset.combine_first,
    :func:`~grid_process.timeseries.setindex_validtime`

    Note
    ----
    This function is not aimed to combine models that do not have the same grid
    and members, but to combine them after regridding/interpolation and ensemble
    statistic computations (quantiles, probabilities). A typical use case for this
    function could be to combine past deterministic analyses to all members of a NWP
    model as it will automatically broadcast it to the ensemble members dimension and
    pioritize analyses over the NWP forecast for common timestamps in both datasets.
    """

    units_attrs = consistency_check(datasets, check_vars=False)
    datasets = [setindex_validtime(ds) for ds in datasets if ds.sizes[get_t_dim(ds)]]
    combined_attrs = {"source": blend_src_attr(datasets)}
    combined_ds = datasets[0].copy()
    combined_ds = combined_ds.drop_vars(get_reftime(combined_ds, "coord"), errors="ignore")  # type: ignore[arg-type]
    vt_dim = get_validtime(combined_ds)
    vt_start = combined_ds[vt_dim][0].values if strict_start else None

    for ds in datasets[1:]:
        vt_dim_ds = get_validtime(ds)
        if vt_dim_ds != vt_dim:
            ds = ds.rename({vt_dim_ds: vt_dim})
        if vt_start is not None:
            ds = ds.sel({vt_dim: slice(vt_start, None)})
        ds = ds.drop_vars(get_reftime(ds, "coord"), errors="ignore")  # type: ignore[arg-type]
        combined_ds = combined_ds.combine_first(ds)

    replace_vars_attrs(combined_ds, units_attrs)
    combined_ds.attrs = combined_attrs

    return combined_ds


def cleaver(  # pylint: disable=too-many-locals
    ds_tail: xr.Dataset,
    ds_head: xr.Dataset,
    h: int = 6,
    interp_method: InterpMethods = "cubic",
) -> xr.Dataset:
    """
    Combine two timeseries by applying a linear seamless transition from the
    last values of the tail to the remaining of the head.

    Parameters
    ----------
    ds_tail : xarray.Dataset
        dataset with a valid time or lead time dimension. Contains the tail of
        the newly produced timeseries
    ds_head : xarray.Dataset
        dataset with a valid time or lead time dimension. Contains the head of
        the newly produced timeseries
    h : int, optional
        number of hours over which the seamless transition is done. Default is
        6
    interp_method : str
        arg passed directly to
        :func:`~grid_process.timeseries.interptogranularity` in case the
        tail and the head do not overlap. Default is 'cubic'

    Returns
    -------
    seamless_ds : xarray.Dataset
        newly produced seamless timeseries from the two input datasets. Has a
        valid time dimension labelled by numpy.datetime64 although inputs could
        have a lead time dimension (numpy.timedelta64)

    See also
    --------
    :func:`~grid_process.timeseries.interptogranularity`
    :func:`~grid_process.timeseries.setindex_validtime`
    """

    units_attrs = consistency_check((ds_tail, ds_head))
    combined_attrs = {"source": blend_src_attr([ds_tail, ds_head])}
    ds_tail = setindex_validtime(ds_tail)
    ds_tail = ds_tail.drop_vars(get_reftime(ds_tail, "coord"), errors="ignore")  # type: ignore[arg-type]
    ds_head = setindex_validtime(ds_head)
    ds_head = ds_head.drop_vars(get_reftime(ds_head, "coord"), errors="ignore")  # type: ignore[arg-type]

    vt_dim = get_validtime(ds_tail)
    vt_dim_head = get_validtime(ds_head)

    if vt_dim_head != vt_dim:
        ds_head = ds_head.rename({vt_dim_head: vt_dim})

    overlap_t = ds_tail[vt_dim].where(ds_tail[vt_dim].isin(ds_head[vt_dim]), drop=True).values

    if overlap_t.size == 0:
        logger.warning(
            "No seamless transition is performed as no overlapping timestamps was found between the tail and the head"
        )
        combined_ds = ds_tail.combine_first(ds_head)
        combined_ds.attrs = combined_attrs
        return interptogranularity(combined_ds, method=interp_method)

    last_t = overlap_t.max()
    h_td = np.timedelta64(h, "h")
    incr = np.timedelta64(1, "s")
    ds_clever = ds_head.sel({vt_dim: slice(last_t + incr, last_t + h_td - incr)})
    factor = (1 - (ds_clever[vt_dim] - last_t) / h_td).astype(np.float32)
    last_t_pd = pd.to_datetime(last_t)  # hack for bug with current xarray version

    delta_last_t = ds_tail.sel({vt_dim: last_t_pd}) - ds_head.sel({vt_dim: last_t_pd})
    ds_clever += factor * delta_last_t

    seamless_ds = xr.concat(
        (
            ds_tail.sel({vt_dim: slice(None, last_t)}),
            ds_clever,
            ds_head.sel({vt_dim: slice(last_t + h_td, None)}),
        ),
        dim=vt_dim,
    )

    replace_vars_attrs(seamless_ds, units_attrs)
    seamless_ds.attrs = combined_attrs

    return seamless_ds


def blend_src_attr(datasets: Sequence[xr.Dataset]) -> dict[str, str]:
    """
    Returns a dict with keys being the sources of the input datasets and
    values the corresponding reftimes. This dict can become the output
    ``source`` attribute when blending datasets together.
    """

    src_dict = {}
    for ds in datasets:
        reftime_coord = get_reftime(ds, "coord")
        if isinstance(ds.source, dict):
            src_dict.update(ds.source)
        elif reftime_coord is not None:
            if len(ds[reftime_coord].dims) == 0:
                reftime = ds[reftime_coord].values
            else:
                reftime = ds[reftime_coord].values[-1]
            src_dict.update({ds.source: pd.to_datetime(reftime).isoformat()})
        elif ds.source not in src_dict:
            src_dict.update({ds.source: None})

    return src_dict
