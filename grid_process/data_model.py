# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module with utility functions to deal with the numerous supported data models
"""

import logging
import json
from typing import Any, TypeVar, Literal, overload
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
import geopandas as gpd
import xarray as xr

from grid_process import CONFIG

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

DATA_MODEL = CONFIG.data_model
FLAT_DIM: list[str] = DATA_MODEL.grid.unstructured_dim
YX: tuple[tuple[str, str], ...] = tuple(zip(DATA_MODEL.grid.y, DATA_MODEL.grid.x))
LATLON: tuple[tuple[str, str], ...] = tuple(yx for yx in YX if "lat" in yx[0] and "lon" in yx[1])
SURF_ALT: list[str] = DATA_MODEL.grid.surf_alt
REFTIME: list[str] = DATA_MODEL.reftime
VALIDTIME: list[str] = DATA_MODEL.validtime
LEADTIME: list[str] = DATA_MODEL.leadtime
ENS_DIM: list[str] = DATA_MODEL.ens_dim
POI_ALT_COL: list[str] = DATA_MODEL.poi.alt_from


def get_t_dim(ds: xr.Dataset) -> str:
    """
    Returns name of temporal dimension according to the possible options defined
    in the config (could be overwritten by setting appropriate env variable).
    Temporal dimension could be expressed as valid time or lead time.
    """

    validtime_dim = get_validtime(ds)
    if validtime_dim is not None:
        return validtime_dim

    leadtime_dim = get_leadtime(ds)
    if leadtime_dim is not None:
        return leadtime_dim

    raise ValueError(
        f"Temporal dimension must either be one of {VALIDTIME} (valid time) or "
        f"{LEADTIME} (lead time) or have an adequate CF standard_name attribute"
    )


def get_temporal_dims(
    data: xr.Dataset | xr.DataArray,
) -> tuple[str | None, str | None, str | None]:
    """
    Returns names of temporal dimensions according to the possible options defined
    in the config (could be overwritten by setting appropriate env variable) or
    according to CF standard name. The returned tuple contains dimension names in
    the following order: (reftime, leadtime, validtime).
    """

    return _get_temporal(data, "dim")


def get_temporal_coords(
    data: xr.Dataset | xr.DataArray,
) -> tuple[str | None, str | None, str | None]:
    """
    Returns names of temporal coordinates according to the possible options defined
    in the config (could be overwritten by setting appropriate env variable) or
    according to CF standard name. The returned tuple contains coordinate names in
    the following order: (reftime, leadtime, validtime).
    """

    return _get_temporal(data, "coord")


def _get_temporal(
    data: xr.Dataset | xr.DataArray, kind: Literal["dim", "coord"]
) -> tuple[str | None, str | None, str | None]:
    """
    Returns names of temporal dimensions or coordinates.
    """

    reftime = get_reftime(data, kind)
    leadtime = get_leadtime(data, kind)
    validtime = get_validtime(data, kind)

    return reftime, leadtime, validtime


def get_reftime(data: xr.Dataset | xr.DataArray, kind: Literal["dim", "coord"] = "dim") -> str | None:
    """
    Returns name of forecast reference time dimension or coordinate according to
    the possible options defined in the config (could be overwritten by setting
    appropriate env variable) or according to CF standard name.
    """

    return _get_supported_dimcoord(data, kind, REFTIME, "forecast_reference_time")


@overload
def get_validtime(
    data: xr.Dataset | xr.DataArray,
    kind: Literal["dim", "coord"] = "dim",
    raise_notfound: Literal[True] = True,
) -> str: ...
@overload
def get_validtime(
    data: xr.Dataset | xr.DataArray,
    kind: Literal["dim", "coord"] = "dim",
    raise_notfound: Literal[False] = False,
) -> str | None: ...


def get_validtime(
    data: xr.Dataset | xr.DataArray,
    kind: Literal["dim", "coord"] = "dim",
    raise_notfound: bool = False,
) -> str | None:
    """
    Returns name of valid time dimension or coordinate according to the possible
    options defined in the config (could be overwritten by setting appropriate
    env variable) or according to CF standard name. When raise_notfound is set
    to True, raise a ValueError if a valid time dimension/coordinate can not be
    found.
    """

    validtime = _get_supported_dimcoord(data, kind, VALIDTIME, "time")
    if raise_notfound and validtime is None:
        raise ValueError(
            f"Could not find a valid time dimension. Must be one of {VALIDTIME}"
            " or have the CF standard_name attribute 'time'"
        )

    return validtime


def get_leadtime(data: xr.Dataset | xr.DataArray, kind: Literal["dim", "coord"] = "dim") -> str | None:
    """
    Returns name of lead time dimension or coordinate according to the possible
    options defined in the config (could be overwritten by setting appropriate
    env variable) or according to CF standard name.
    """

    return _get_supported_dimcoord(data, kind, LEADTIME, "forecast_period")


def _get_supported_dimcoord(
    data: xr.Dataset | xr.DataArray,
    kind: Literal["dim", "coord"],
    config_options: list[str],
    cf_standard: str,
) -> str | None:
    """
    Returns name of existing dimension or coordinate in ``data`` from the
    ``config_options`` list or with a CF conventions standard_name attribute
    ``cf_standard``.
    """

    dim_coord: tuple[str, ...] = tuple(data.coords) if kind == "coord" else tuple(data.dims)  # type: ignore[arg-type]
    for name in config_options:
        if name in dim_coord:
            return name

    for name in dim_coord:
        if data[name].attrs.get("standard_name", "").lower() == cf_standard:
            return name

    return None


def get_ensemble_dim(data: xr.Dataset | xr.DataArray) -> str:
    """
    Returns name of ensemble members dimension according to the possible options
    defined in the config (could be overwritten by setting appropriate env variable)
    or according to CF standard name.
    """

    ens_dim = _get_supported_dimcoord(data, "dim", ENS_DIM, "realization")
    if ens_dim is None:
        raise ValueError(
            "Could not find an ensemble members dimension. Must be one of "
            f"{ENS_DIM} or have the CF standard_name attribute 'realization'"
        )

    return ens_dim


def set_grid_mapping(ds: xr.Dataset) -> xr.Dataset:
    """
    Set grid_mapping attribute according to internal conventions when the input
    grid follow CF conventions (grid mapping encoded as a data variable). The
    data variable encoding the grid mapping is removed.
    """

    def serializable_mapping() -> dict[str, Any]:
        return {
            key: float(value) if isinstance(value, np.floating) else value
            for key, value in ds_out[mapname].attrs.items()
        }

    ds_out = ds.copy(deep=False)
    grid_map = set()
    for varname, da in ds_out.items():
        if "grid_mapping" in da.attrs:
            grid_map.add(da.attrs.pop("grid_mapping"))
        elif "grid_mapping_name" in da.attrs:
            grid_map.add(varname)

    if len(grid_map) == 0:
        return ds_out
    if len(grid_map) > 1:
        raise ValueError("Several grid mappings in the same dataset is not supported")

    mapname = grid_map.pop()
    if {"grid_mapping", "crs"}.isdisjoint(ds_out.attrs) and mapname in ds_out:
        ds_out.attrs["grid_mapping"] = json.dumps(serializable_mapping())

    return ds_out.drop_vars(mapname, errors="ignore")


def get_spatial_dims(grid: xr.Dataset | xr.DataArray) -> tuple[str, ...]:
    """
    Returns names of spatial dimensions according to the possible options defined
    in the config (could be overwritten by setting appropriate env variable)
    or according to CF conventions. The returned tuple has size one in case of
    unstructured grid otherwise contains dimension names in the order (y, x).
    """

    flat_dim = get_flat_dim(grid)
    if flat_dim is not None:
        return (flat_dim,)

    spatial_dims = _get_spatial(grid, tuple(grid.dims))  # type: ignore[arg-type]

    if None in spatial_dims:
        raise ValueError(
            f"Spatial dimensions must either be one of {YX}, have CF conventions"
            f" attributes for arbitrary names or be one of {FLAT_DIM} in case of"
            " unstructured grid"
        )

    return spatial_dims  # type: ignore[return-value]


def get_spatial_coords(grid: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    """
    Returns names of non-dimension spatial coordinates according to the
    possible options defined in the config (could be overwritten by setting
    appropriate env variable) or according to CF conventions. The returned
    tuple contains coordinate names in the order (y, x).
    """

    notdim_coords = tuple(set(grid.coords) - set(grid.dims))
    spatial_coords = _get_spatial(grid, notdim_coords)  # type: ignore[arg-type]

    if None in spatial_coords:
        raise ValueError(
            "Could not find non-dimension spatial coordinates. Must be one of "
            f"{YX} or have CF conventions attributes for arbitrary names"
        )

    return spatial_coords  # type: ignore[return-value]


def _get_spatial(grid: xr.Dataset | xr.DataArray, dim_or_coord: tuple[str, ...]) -> tuple[str | None, str | None]:
    """
    Returns names of spatial dimensions or coordinates in ``dim_or_coord``.
    """

    for y, x in YX:
        if y in dim_or_coord and x in dim_or_coord:
            return y, x

    # look for coordinates with CF attributes in coords
    spatial: list[str | None] = [None, None]
    for name in dim_or_coord:
        attrs = grid[name].attrs
        if (attrs.get("axis", "").upper() == "Y") or (
            attrs.get("standard_name", "").lower() in ("latitude", "grid_latitude", "projection_y_coordinate")
        ):
            spatial[0] = name
        elif (attrs.get("axis", "").upper() == "X") or (
            attrs.get("standard_name", "").lower() in ("longitude", "grid_longitude", "projection_x_coordinate")
        ):
            spatial[1] = name

        if None not in spatial:
            break

    return (spatial[0], spatial[1])


def get_flat_dim(grid: xr.Dataset | xr.DataArray) -> str | None:
    """
    Returns spatial dimension name for an unstructured grid according to the
    possible options defined in the config (could be overwritten by setting
    appropriate env variable).
    """

    for dim in FLAT_DIM:
        if dim in grid.dims:
            return dim

    return None


def get_latlon_coords(grid: xr.Dataset | xr.DataArray, nondim: bool = True) -> tuple[str, str] | None:
    """
    Returns names of latitude and longitude coordinates if they exist according
    to the possible options defined in the config (could be overwritten by setting
    appropriate env variable) or according to CF conventions. The returned tuple
    contains coordinate names in the order (latitude, longitude). When ``nondim``
    is True (default), only checks coordinates that are non-dimensional.
    """

    coords = tuple(grid.coords)
    if nondim:
        coords = tuple(set(coords) - set(grid.dims))
    return _get_latlon(grid, coords)  # type: ignore[arg-type]


def get_latlon_dims(grid: xr.Dataset | xr.DataArray) -> tuple[str, str] | None:
    """
    Returns names of latitude and longitude dimensions if they exist according
    to the possible options defined in the config (could be overwritten by setting
    appropriate env variable) or according to CF conventions. The returned tuple
    contains coordinate names in the order (latitude, longitude).
    """

    return _get_latlon(grid, tuple(grid.dims))  # type: ignore[arg-type]


def _get_latlon(grid: xr.Dataset | xr.DataArray, dim_or_coord: tuple[str, ...]) -> tuple[str, str] | None:
    """
    Returns names of latitude and longitude dimensions or coordinates in
    ``dim_or_coord``.
    """

    for lat, lon in LATLON:
        if lat in dim_or_coord and lon in dim_or_coord:
            return lat, lon

    # look for coordinates with CF attributes in coords
    latlon: list[str | None] = [None, None]
    for name in dim_or_coord:
        coord_attrs = grid[name].attrs
        if coord_attrs.get("standard_name", "") == "latitude" or coord_attrs.get("long_name", "") == "latitude":
            latlon[0] = name
        elif coord_attrs.get("standard_name", "") == "longitude" or coord_attrs.get("long_name", "") == "longitude":
            latlon[1] = name

        if None not in latlon:
            break

    if None in latlon:
        return None

    return (latlon[0], latlon[1])  # type: ignore[return-value]


def get_surf_alt(grid: xr.Dataset | xr.DataArray) -> str:
    """
    Returns name of the surface altitude coordinate variable according to the
    possible options defined in the config (could be overwritten by setting
    appropriate env variable).
    """

    for name in SURF_ALT:
        if name in grid.coords:
            return name

    raise ValueError(f"Missing one of {SURF_ALT} coordinate variable")


def get_poi_alt(poi_info: gpd.GeoDataFrame) -> NDArray[np.float32]:
    """
    Returns POIs altitudes from ``poi_info`` according to the column names defined
    in the config (could be overwritten by setting appropriate env variable) or
    from a "z" geometry. Altitude data is expected to be in meters above sea level.
    """

    for name in POI_ALT_COL:
        if name in poi_info.columns:
            return poi_info[name].values.astype(np.float32)

    if poi_info.iloc[0].geometry.has_z:
        return poi_info.geometry.z.values.astype(np.float32)

    raise ValueError(
        "GeoDataFrame must have a column with one of the following names "
        f"{POI_ALT_COL} or a z coordinate. Data is assumed to be in meters"
    )


def get_grid_crs(grid: xr.Dataset | xr.DataArray) -> CRS:
    """
    Get correct CRS according to supported conventions.

    Parameters
    ----------
    grid : xarray Dataset or DataArray
        contains spatial horizontal coordinates. Must have a "grid_mapping" or "crs"
        attribute referencing the spatial dimensions unless `longitude` and
        `latitude` (or configured equivalents) are dimensions.

    Returns
    -------
    pyproj CRS :
        pyproj representation of input grid coordinate reference system

    See also
    --------
    :func:`~grid_process.helpers.set_grid_mapping`
    """

    if "crs" in grid.attrs:
        return CRS.from_user_input(grid.crs)
    if "grid_mapping" in grid.attrs:
        return parse_grid_mapping(grid.grid_mapping)
    if get_latlon_dims(grid) is not None:
        return CRS.from_string("EPSG:4326")

    raise ValueError("The dataset must have a grid_mapping or crs attribute or have latitude and longitude dimensions")


def parse_grid_mapping(grid_mapping: str | dict[str, Any]) -> CRS:
    """
    Parse internal grid database style "grid_mapping" attribute and decode it with Pyproj.

    Parameters
    ----------
    grid_mapping : dict or dict encapsulated in a string
        contains CF conventions attributes parsable by pyproj.CRS.from_cf() or
        the key "epsg_code" with value being an authority string (e.g. EPSG:4326)

    Returns
    -------
    pyproj CRS :
        pyproj representation of input grid_mapping

    See also
    --------
    :func:`get_grid_crs`
    """

    if isinstance(grid_mapping, str):
        grid_mapping = json.loads(grid_mapping)
    if "epsg_code" in grid_mapping:
        return CRS.from_string(grid_mapping["epsg_code"])  # type: ignore[index]

    return CRS.from_cf(grid_mapping)  # type: ignore[arg-type]


def get_coords_name_crs(
    grid: xr.Dataset | xr.DataArray, spatial_dims: tuple[str, ...] | None = None
) -> tuple[str, str, CRS]:
    """
    Get correct coordinate names and CRS according to supported conventions. When
    non-dimensional longitude and latitude coordinates arrays are present, these
    are used instead of the native spatial dimension coordinates.

    Parameters
    ----------
    grid : xarray Dataset or DataArray
        contains spatial horizontal coordinates. Must have a "grid_mapping" or "crs"
        attribute referencing the spatial dimensions unless longitude and
        latitude coordinates are present.
    spatial_dims : tuple of string(s), optional
        name of the ``grid`` spatial dimension(s). Has size one in case of
        unstructured grid otherwise contains dimension names in the order (y, x).
        If not provided, calls :func:`~grid_process.helpers.get_spatial_dims`

    Returns
    -------
    grid_y : str
        name of the northward coordinate. Usually 'x' or 'longitude'
    grid_x : str
        name of the eastward coordinate. Usually 'y' or 'latitude''
    grid_crs : pyproj CRS
        pyproj representation of input grid coordinate reference system
    """

    latlon = get_latlon_coords(grid)
    if latlon is not None:
        grid_y, grid_x = latlon
        return grid_y, grid_x, CRS.from_string("EPSG:4326")

    if spatial_dims is None:
        spatial_dims = get_spatial_dims(grid)

    if len(spatial_dims) == 1:
        grid_y, grid_x = get_spatial_coords(grid)
        return grid_y, grid_x, get_grid_crs(grid)

    grid_y, grid_x = spatial_dims
    if grid_y in grid.coords and grid_x in grid.coords:
        return grid_y, grid_x, get_grid_crs(grid)

    raise ValueError("Missing spatial coordinates")


def is_decreasing1d(coords: xr.DataArray) -> bool:
    """
    Whether 1 dimensional ``coords`` are in decreasing order.
    """

    return bool(np.ediff1d(coords.values[:2])[0] < 0)


def is_decreasing(coords: xr.DataArray, dim: str) -> bool:
    """
    Checks whether ``coords`` are in decreasing order along dimension ``dim``.
    """

    if coords.ndim > 1:
        other_dim = next(d for d in coords.dims if d != dim)
        mid_index = coords.sizes[other_dim] // 2
        return is_decreasing1d(coords.isel({other_dim: mid_index}))

    return is_decreasing1d(coords)
