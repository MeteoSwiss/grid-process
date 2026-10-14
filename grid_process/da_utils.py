# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module with general utility functions that operate on xarray.DataArray
"""

import logging
from typing import Any
from collections.abc import Sequence, Hashable
from functools import partial

import xarray as xr
import numpy as np

from grid_process.helpers import make_accessor, update_vars_attrs

logger = logging.getLogger(__name__)


# Register xarray accessor so module functions can also be called as methods
# on xr.DataArray objects
@xr.register_dataarray_accessor("da_utils")
class DaUtils:
    """
    Xarray accessors class for general utility functions on DataArray
    """

    def __init__(self, da: xr.DataArray):
        self._data = da


dataarray_accessor = partial(make_accessor, accessor_class=DaUtils)


@dataarray_accessor
def da_to_ds(da: xr.DataArray, vars_attrs: dict[Hashable, dict[Hashable, Any]]) -> xr.Dataset:
    """
    Transform a dataarray, containing model data and one or several
    variables specified by the `variable` dimension, to a dataset. The function
    also reassign the attributes of the data variables saved in ``vars_attrs``.

    Parameters
    ----------
    da : xarray.DataArray
        contains one or more variables specified in the `variable` dimension
    vars_attrs : dict
        contains the attributes of each variables found along the `variable`
        dimension

    Returns
    -------
    ds : xarray.Dataset
        the dataset is composed of the data variables of the input DataArray

    See also
    --------
    :func:`~grid_process.ds_utils.ds_to_da` which is the inverse function
    """

    ds = da.to_dataset("variable")
    update_vars_attrs(ds, vars_attrs)

    return ds


@dataarray_accessor
def bin_to_class(
    da: xr.DataArray,
    bins: Sequence[float],
    bins_label: Sequence[str | int],
    da_name: str | None = None,
) -> xr.DataArray:
    """
    Convert numerical values into classes as defined by ``bins`` and
    ``bins_label``.

    Parameters:
    -----------
    da : xarray.DataArray
        array with numerical values
    bins : sequence of floats
        defines the boudaries for the classes
    bins_label : sequence of str or int
        class labels. len(``bin_labels``) = len(``bins``) + 1
    da_name : str, optional
        used to rename the output DataArray. By default no renaming is done.

    Returns
    -------
    class_da : xarray.DataArray
        same as input array but with data converted from numerical values to
        classes
    """

    bins_label_arr = np.asarray(bins_label)
    class_da = xr.apply_ufunc(lambda x: bins_label_arr[np.digitize(x, bins=bins, right=True)], da)

    if da_name is not None:
        class_da = class_da.rename(da_name)
    return class_da
