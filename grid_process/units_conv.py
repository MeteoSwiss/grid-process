# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module to control, parse and convert units
"""

import logging
import re
from functools import partial
from collections.abc import Sequence, Hashable

import numpy as np
import xarray as xr
from pint import UnitRegistry

from grid_process.helpers import make_accessor

logger = logging.getLogger(__name__)


class UnitsError(Exception):
    """
    Class to raise units errors
    """


# Register xarray accessor so module functions can also be called as methods
# on xr.DataArray objects
@xr.register_dataarray_accessor("units_conv")
class DaUnits:
    """
    Xarray accessors class for units utility functions on DataArray
    """

    def __init__(self, da: xr.DataArray):
        self._data = da


@xr.register_dataset_accessor("units_conv")
class DsUnits:
    """
    Xarray accessors class for units utility functions on Dataset
    """

    def __init__(self, ds: xr.Dataset):
        self._data = ds


dataarray_accessor = partial(make_accessor, accessor_class=DaUnits)
dataset_accessor = partial(make_accessor, accessor_class=DsUnits)

ureg: UnitRegistry = UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    preprocessors=[
        partial(
            re.sub,
            r"(?<=[A-Za-z])(?![A-Za-z])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])",
            "**",
        ),
        partial(re.sub, r"\b[h]\b", "hour"),  # avoid parsing of Plank constant for h
    ],
)


@dataarray_accessor
def da_to(da: xr.DataArray, dst_units: str) -> xr.DataArray:
    """
    Convert units in a xarray.DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        with attribute "units"
    dst_units : str
        destination units

    Returns
    -------
    da_out : xarray.DataArray
        contains the data array with converted units

    Raises
    ------
    pint.errors.DimensionalityError
        if the conversion is not possible
    """

    da = da.copy()
    src_units_parsed = ureg.parse_units(da.units)
    dst_units_parsed = ureg.parse_units(dst_units)

    if src_units_parsed == dst_units_parsed:
        pass
    else:
        da_quantity = da.values * src_units_parsed
        da.values = da_quantity.to(dst_units_parsed).magnitude

    da.attrs["units"] = dst_units
    return da


@dataset_accessor
def ds_to(ds: xr.Dataset, dst_units: str | dict[Hashable, str]) -> xr.Dataset:
    """
    Convert units in a xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        with variable attributes "units"
    dst_units : str or dict
        destination units for each data variable to be converted.
        Variable names not specified in ``dst_units`` are kept in the same units.
        If passed as a single string, all variables in the dataset are converted to the
        same units

    Returns
    -------
    ds_out : xarray.Dataset
        contains the dataset with converted units

    Raises
    ------
    pint.errors.DimensionalityError
        if the conversion is not possible
    """

    if isinstance(dst_units, str):
        dst_units = {var: dst_units for var in ds}
    ds_out = ds.copy()
    for var, dst_unit in dst_units.items():
        ds_out[var] = da_to(ds[var], dst_unit)
    return ds_out


def consistency_check(datasets: Sequence[xr.Dataset], check_vars: bool = True) -> dict[str, dict[str, str]]:
    """
    Checks that the input datasets have same data variables and units. If it is
    the case, returns the 'units' attributes of each data variable otherwise
    raises an error.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        each data variable in the datasets must have a 'units' attributes
    check_vars : bool, optional
        if False, allows datasets to have different variables, but common ones
        must have same units. Default is True

    Returns
    -------
    out : dict
        contains the 'units' attribute of the data variables in the input
        datasets. In case same units are specified in various format, the
        dominant format is kept if there is one otherwise the format from the
        first dataset in the input sequence

    Raises
    ------
    UnitsError
        if same variables across the input datasets have different units
    ValueError
        when ``check_vars=True`` and the input datasets do not have exact same
        set of variables
    """

    vars_attrs = np.array([{varname: da.units for varname, da in ds.data_vars.items()} for ds in datasets])

    if check_vars:
        if not np.all([dic.keys() == vars_attrs[0].keys() for dic in vars_attrs]):
            raise ValueError("The datasets to combine must have matching variables, which was not the case")

    unique = {key: {"units": value} for elem in vars_attrs[::-1] for key, value in elem.items()}
    if np.all(
        [
            ureg.parse_units(value) == ureg.parse_units(unique[key]["units"])
            for dic in vars_attrs
            for key, value in dic.items()
        ]
    ):
        return unique

    raise UnitsError("Same variables among the datasets to combine have different units. Please convert units first")


def units_check(datasets: Sequence[xr.Dataset]) -> None:
    """
    Check that the data variables in the datasets all have same single units.
    If not, it raises an error.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        each data variable in the datasets must have a 'units' attributes

    Raises
    ------
    UnitsError
        if any variable in any of the input datasets have different units
    """

    units = [ureg.parse_units(da.units) for ds in datasets for _, da in ds.data_vars.items()]

    if len(set(units)) != 1:
        raise UnitsError("Variables in the datasets do not all have same single units")
