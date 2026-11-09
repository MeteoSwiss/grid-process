# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module for general helper functions and decorators
"""

import logging
import time
from functools import wraps
from typing import Any, overload
from collections.abc import Callable, Hashable

import xarray as xr
import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer, CRS

logger = logging.getLogger(__name__)


def timefn(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Generic decorator to log function processing times
    """

    @wraps(func)
    def measure_time(*args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        result = func(*args, **kwargs)
        logger.debug("@timefn: %s took %.3f sec", func.__name__, time.monotonic() - t0)
        return result

    return measure_time


@overload
def make_accessor(func: Callable[..., xr.Dataset], accessor_class: type) -> Callable[..., xr.Dataset]: ...
@overload
def make_accessor(func: Callable[..., xr.DataArray], accessor_class: type) -> Callable[..., xr.DataArray]: ...


def make_accessor(
    func: Callable[..., xr.Dataset | xr.DataArray], accessor_class: type
) -> Callable[..., xr.Dataset | xr.DataArray]:
    """
    Generic decorator to turn functions operating on xarray objects into
    accessors based on a registered dataset or dataarray ``accessor_class``
    """

    @wraps(func)
    def method(accessor: xr.Dataset | xr.DataArray, *args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
        return func(accessor._data, *args, **kwargs)  # pylint: disable=protected-access  # own accessor classes

    setattr(accessor_class, func.__name__, method)
    return func


def keepattrs(func: Callable[..., xr.Dataset]) -> Callable[..., xr.Dataset]:
    """
    Decorator to keep dataset and data variables attributes for functions
    operating on xarray.Datasets
    """

    @wraps(func)
    def wrapper(*args: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        vars_attrs = {name: da.attrs for name, da in args[0].data_vars.items()}
        ds_attrs = args[0].attrs
        ds_out = func(*args, **kwargs)
        update_vars_attrs(ds_out, vars_attrs)
        ds_out.attrs.update(ds_attrs)
        return ds_out

    return wrapper


def update_vars_attrs(ds: xr.Dataset, vars_attrs: dict[Hashable, dict[Hashable, Any]]) -> None:
    """
    Updates data variables attributes in a xarray.Dataset. The operation is
    done "in-place".

    Parameters
    ----------
    ds : xarray.Dataset
        can contain one or several data variables
    vars_attrs : dict
        contains the attributes of each data variables in the input dataset
    """

    for varname, attributes in vars_attrs.items():
        ds[varname].attrs.update(attributes)


def replace_vars_attrs(ds: xr.Dataset, vars_attrs: dict[str, dict]) -> None:
    """
    Replaces data variables attributes in a xarray.Dataset. The operation is
    done "in-place".

    Parameters
    ----------
    ds : xarray.Dataset
        can contain one or several data variables
    vars_attrs : dict
        contains the new attributes of each data variables in the input dataset
        to be replaced
    """

    for varname, attributes in vars_attrs.items():
        ds[varname].attrs = attributes


@overload
def reproject(
    x_coords: NDArray[np.floating],
    y_coords: NDArray[np.floating],
    src_crs: CRS | str,
    dst_crs: CRS | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def reproject(
    x_coords: list[float], y_coords: list[float], src_crs: CRS | str, dst_crs: CRS | str
) -> tuple[list[float], list[float]]: ...
@overload
def reproject(
    x_coords: tuple[float, ...],
    y_coords: tuple[float, ...],
    src_crs: CRS | str,
    dst_crs: CRS | str,
) -> tuple[tuple[float, ...], tuple[float, ...]]: ...


def reproject(
    x_coords: NDArray[np.floating] | list[float] | tuple[float, ...],
    y_coords: NDArray[np.floating] | list[float] | tuple[float, ...],
    src_crs: CRS | str,
    dst_crs: CRS | str,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[list[float], list[float]]
    | tuple[tuple[float, ...], tuple[float, ...]]
):
    """
    Reprojects `x` and `y` coordinates from a source CRS to a new specified
    one.

    Parameters
    ----------
    x_coords : numpy.ndarray, list or tuple
        contains `x` coordinates.
    y_coords : numpy.ndarray, list or tuple
        contains `y` coordinates
    src_crs : pyproj.CRS of the source or equivalent string
        The value can be anything accepted by
        pyproj.CRS.from_user_input, such as an authority string
        (e.g. "EPSG:4326") or a WKT string
    dst_crs : pyproj.CRS of the destination or equivalent string
        The value can be anything accepted by
        pyproj.CRS.from_user_input, such as an authority string
        (eg "EPSG:4326") or a WKT string

    Returns
    -------
    x_coords_new : numpy.ndarray, list or tuple of float64
        same as input ``x_coords`` in the new CRS and cast as float64
    y_coords_new : numpy.ndarray, list or tuple of float64
        same as input ``y_coords`` in the new CRS and cast as float64

    Note
    ----
    Geographic CRS defined on a sphere without datum (e.g. the rotated pole of a CF grid mapping with
    ``earth_radius``) are considered to be based on WGS84, see :func:`sphere_as_wgs84`. CRS parsed from such a CF
    "grid_mapping" attribute are already based on WGS84 (see :func:`~grid_process.data_model.parse_grid_mapping`),
    this covers the other ones (e.g. "crs" attribute, POIs CRS or direct calls).
    """

    transformer = Transformer.from_crs(sphere_as_wgs84(src_crs), sphere_as_wgs84(dst_crs), always_xy=True)

    return transformer.transform(x_coords, y_coords)


def sphere_as_wgs84(crs: CRS | str) -> CRS:
    """
    Returns a geographic CRS defined on a sphere (e.g. the rotated pole or latitude/longitude of a CF grid mapping
    with ``earth_radius``, as used by NWP models) with WGS84 as base CRS instead, keeping the pole rotation if any.
    Other CRS are returned unchanged.

    The latitudes and longitudes on the sphere of NWP models are usually considered as WGS84 latitudes and
    longitudes (e.g. the 2D ``latitude`` and ``longitude`` coordinates of their output). Without a defined datum,
    pyproj transforms them to other datums with a "ballpark" transformation that leaves out the datum shift, e.g.
    ~180 m in Switzerland between a rotated pole grid and Swiss LV95 coordinates. With WGS84 as base CRS, the datum
    shift is applied as for WGS84 latitudes and longitudes. The rotation of the pole is the same on any ellipsoid.

    Parameters
    ----------
    crs : pyproj.CRS or equivalent string
        anything accepted by pyproj.CRS.from_user_input

    Returns
    -------
    pyproj.CRS
    """

    crs = CRS.from_user_input(crs)
    ellipsoid = crs.ellipsoid
    if not crs.is_geographic or ellipsoid is None or ellipsoid.semi_major_metre != ellipsoid.semi_minor_metre:
        return crs

    wgs84 = CRS.from_epsg(4326)
    crs_json = crs.to_json_dict()
    if "base_crs" not in crs_json:  # latitude and longitude on the sphere
        return wgs84

    # derived geographic CRS, e.g. rotated pole: same conversion (rotation), WGS84 as base CRS
    crs_json["base_crs"] = wgs84.to_json_dict()
    return CRS.from_json_dict(crs_json)


def outer_ring(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Coordinates of the outer ring of points of a structured grid, from 1D (dimension) or 2D coordinates in the
    order (y, x). On a grid, it encloses the image of the whole grid in any continuous projection, e.g. the bounding
    box of a reprojected grid is the bounding box of its reprojected outer ring.

    Parameters
    ----------
    x_coords, y_coords : numpy.ndarray
        1D coordinates of the x and y dimensions, or 2D coordinates of shape (y, x)

    Returns
    -------
    x_ring, y_ring : numpy.ndarray
        1D coordinates of the outer ring points (first and last rows, then first and last columns)
    """

    if x_coords.ndim == 1:
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)

    def ring(coords: np.ndarray) -> np.ndarray:
        return np.append(coords[[0, -1], :].ravel(), coords[1:-1, [0, -1]].ravel())

    return ring(x_coords), ring(y_coords)
