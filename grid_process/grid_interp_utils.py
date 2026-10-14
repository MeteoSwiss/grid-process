# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module with utilities for spatial interpolation of gridded datasets
"""

from typing import Any

import numpy as np
from pyproj import CRS
import xarray as xr

from grid_process.helpers import reproject
from grid_process import data_model as dm


Y_DIM, X_DIM = dm.YX[0]
LAT, LON = dm.LATLON[0]


def create_dst_grid(  # pylint: disable=too-many-locals
    dst_grid_crs: Any,
    bottomleft: tuple[float, float],
    topright: tuple[float, float] | None = None,
    nrowcol: tuple[int, int] | None = None,
    resolution: float | None = None,
    add_latlon: bool = False,
    dtype: np.dtype = np.dtype("float32"),
) -> xr.Dataset:
    """
    Create a destination grid on which data can be interpolated. The output
    can be directly used with :func:`interp2grid`. Two out of the three optional
    input arguments ``topright``, ``nrowcol``, ``resolution`` must be provided.

    Parameters
    ----------
    dst_grid_crs : string, int, dict or a CRS object
        Anything accepted by pyproj.crs.CRS.from_user_input:
        https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_user_input
        (e.g. EPSG:4326)
    bottomleft : tuple of floats
        (y, x) coordinates of the center of the bottom left corner pixel
    topright : tuple of floats, optional
        (y, x) coordinates of the center of the top right corner pixel.
        If provided with ``nrowcol``, this is the exact coordinates. If provided
        with ``resolution`` this is a closed upper bound. The exact coordinates of
        the top right corner pixel will then be a multiple of the resolution with
        respect to the bottom left corner.
    nrowcol : tuple of int, optional
        number of rows (y) and columns (x) respectively
    resolution : float, optional
        resolution of the destination grid in the units of the ``grid_mapping``
    add_latlon : bool
        If True, add latitude and longitude 2d arrays as coordinates
    dtype : numpy.dtype
        The dtype in which spatial coordinates are returned

    Returns
    -------
    dst_grid : xarray.Dataset
        dataset with spatial horizontal coordinates and an associated crs attribute

    See also
    --------
    :func:`interp2grid`
    """

    start_y, start_x = bottomleft

    if topright is not None and nrowcol is not None:
        end_y, end_x = topright
        nrows, ncols = nrowcol
        y_coords = np.linspace(start_y, end_y, nrows, dtype=dtype)
        x_coords = np.linspace(start_x, end_x, ncols, dtype=dtype)
    elif topright is not None and resolution is not None:
        end_y, end_x = topright
        y_coords = np.linspace(start_y, end_y, num=int((end_y - start_y) / resolution) + 1, dtype=dtype)
        x_coords = np.linspace(start_x, end_x, num=int((end_x - start_x) / resolution) + 1, dtype=dtype)
    elif nrowcol is not None and resolution is not None:
        nrows, ncols = nrowcol
        end_y = start_y + (nrows - 1) * resolution
        end_x = start_x + (ncols - 1) * resolution
        y_coords = np.linspace(start_y, end_y, nrows, dtype=dtype)
        x_coords = np.linspace(start_x, end_x, ncols, dtype=dtype)
    else:
        raise ValueError("At least two of 'topright', 'nrowcol', 'resolution' arguments must be provided")

    dst_grid = xr.Dataset(
        coords={X_DIM: x_coords, Y_DIM: y_coords},
        attrs={"crs": dst_grid_crs},
    )

    if add_latlon:
        x_2d, y_2d = np.meshgrid(x_coords, y_coords)
        dst_crs = CRS.from_user_input(dst_grid_crs)
        longitude, latitude = reproject(x_2d, y_2d, dst_crs, "EPSG:4326")
        dst_grid = dst_grid.assign_coords(
            {
                LON: ((Y_DIM, X_DIM), longitude.astype(dtype, copy=False)),
                LAT: ((Y_DIM, X_DIM), latitude.astype(dtype, copy=False)),
            }
        )

    return dst_grid


def create_dst_grid_from_src(  # pylint: disable=too-many-locals
    src_grid: xr.Dataset,
    dst_grid_crs: Any,
    add_latlon: bool = False,
) -> xr.Dataset:
    """
    Creates a destination grid on which data can be interpolated from a source grid
    in a different Coordinate Reference System. Domain and sizes are preserved.
    The output can be directly used with :func:`interp2grid`.

    Parameters
    ----------
    src_grid : xarray Dataset
        The source grid from which a new destination grid with same sizes and domain
        is derived. Supported data models include:

          - CF-convention grids (see https://cfconventions.org/)
          - internal MCH models
          - Unstructured grids

        Typical spatial dimension and coordinate names are supported, and can be extended via environment variables:

          - Y coordinate: export GRID_PROCESS__data_model__grid__y (e.g., '["y", "latitude", "lat", "dummy_y"]')
          - X coordinate: export GRID_PROCESS__data_model__grid__x (e.g., '["x", "longitude", "lon", "dummy_x"]')
          - Unstructured dimension: export GRID_PROCESS__data_model__grid__unstructured_dim (e.g., '["dummy_cell"]')
          - When the spatial coordinates variables have a CF compliant `standard_name` attribute, arbitrary names can be
            used directly.

        Spatial CRS must be referenced via one of:

          - Global attribute 'crs' (e.g., "EPSG:2056" or anything accepted by :meth:`pyproj.CRS.from_user_input`)
          - Global attribute 'grid_mapping' (internal grid database convention)
          - CF-convention grid mapping variable
          - If coordinates are named 'latitude'/'longitude' or 'lat'/'lon', WGS84 (EPSG:4326) is assumed.
    dst_grid_crs : string, int, dict or a CRS object
        Anything accepted by pyproj.crs.CRS.from_user_input:
        https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_user_input
        (e.g. EPSG:4326)
    add_latlon : bool
        If True, add latitude and longitude 2d arrays as coordinates

    Returns
    -------
    dst_grid : xarray.Dataset
        dataset with spatial horizontal coordinates and an associated crs attribute

    See also
    --------
    :func:`interp2grid`
    """

    spatial_dims = dm.get_spatial_dims(src_grid)
    src_grid = dm.set_grid_mapping(src_grid)
    src_y, src_x, src_crs = dm.get_coords_name_crs(src_grid, spatial_dims)
    dst_crs = CRS.from_user_input(dst_grid_crs)
    x_min, x_max = src_grid[src_x].values.min(), src_grid[src_x].values.max()
    y_min, y_max = src_grid[src_y].values.min(), src_grid[src_y].values.max()
    x_bnds = (x_min, x_min, x_max, x_max)
    y_bnds = (y_min, y_max, y_min, y_max)
    x_bnds, y_bnds = reproject(x_bnds, y_bnds, src_crs, dst_crs)
    x_bnds, y_bnds = (min(x_bnds), max(x_bnds)), (min(y_bnds), max(y_bnds))

    if len(spatial_dims) == 1:  # need to infer expected sizes for x and y dims
        xy_ratio = (x_bnds[1] - x_bnds[0]) / (y_bnds[1] - y_bnds[0])
        dst_x_size = round(np.sqrt(src_grid.sizes[spatial_dims[0]] * xy_ratio))
        dst_y_size = round(np.sqrt(src_grid.sizes[spatial_dims[0]] / xy_ratio))
        bool_decrease = False
    else:
        dst_x_size, dst_y_size = (
            src_grid.sizes[spatial_dims[1]],
            src_grid.sizes[spatial_dims[0]],
        )
        bool_decrease = dm.is_decreasing(src_grid[src_y], spatial_dims[0])

    src_x_dtype, src_y_dtype = src_grid[src_x].dtype, src_grid[src_y].dtype
    dst_x_dtype = src_x_dtype if src_x_dtype == np.float64 else np.float32
    dst_y_dtype = src_y_dtype if src_y_dtype == np.float64 else np.float32
    dst_x_coords = np.linspace(x_bnds[0], x_bnds[1], dst_x_size, dtype=dst_x_dtype)
    dst_y_coords = np.linspace(y_bnds[0], y_bnds[1], dst_y_size, dtype=dst_y_dtype)
    if bool_decrease:
        dst_y_coords = dst_y_coords[::-1]

    dst_grid = xr.Dataset(
        coords={X_DIM: dst_x_coords, Y_DIM: dst_y_coords},
        attrs={"crs": dst_grid_crs},
    )

    if add_latlon:
        x_2d, y_2d = np.meshgrid(dst_x_coords, dst_y_coords)
        longitude, latitude = reproject(x_2d, y_2d, dst_crs, "EPSG:4326")
        dst_grid = dst_grid.assign_coords(
            {
                LON: ((Y_DIM, X_DIM), longitude.astype(dst_x_dtype, copy=False)),
                LAT: ((Y_DIM, X_DIM), latitude.astype(dst_y_dtype, copy=False)),
            }
        )

    return dst_grid
