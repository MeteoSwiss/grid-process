# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module with general utility functions that operate on xarray.Dataset
"""

import logging
from typing import Any
from collections.abc import Hashable

import xarray as xr

logger = logging.getLogger(__name__)


def ds_to_da(
    ds: xr.Dataset,
) -> tuple[xr.DataArray, dict[Hashable, dict[Hashable, Any]]]:
    """
    Transform a dataset containing model data and one or several data
    variables to a dataarray with the variables stacked as a new dimension.
    The function also returns the attributes of the data variables for further
    use.

    Parameters
    ----------
    ds : xarray.Dataset
        contain one or more variables

    Returns
    -------
    da : xarray.DataArray
        contains the data variables from the input dataset stacked in a new
        `variable` dimension
    vars_attrs : dictionary
        contains the attributes of each data variables in the input dataset.

    See also
    --------
    :func:`~grid_process.da_utils.da_to_ds` which is the inverse function
    """

    vars_attrs = {varname: da.attrs for varname, da in ds.data_vars.items()}
    da = ds.to_array()
    return da, vars_attrs


def cast_dtype(ds_out: xr.Dataset, ds_in: xr.Dataset) -> None:
    """
    Cast back each data variable in the output dataset to the original dtype
    of the input dataset. The operation is done in-place.
    """

    for varname, da in ds_out.items():
        dtype_in = ds_in[varname].dtype
        if da.dtype != dtype_in:
            ds_out[varname] = da.astype(dtype_in)
