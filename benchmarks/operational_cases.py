# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Benchmark of grid-process on its operational cases at MeteoSwiss (see AGENTS.md).

The cases are the interpolations of the data4web pipelines (``interp2poi`` to about 5 600 POIs) and of the seamless
weather provider (``slice_coarsen`` then ``interp2grid`` to a Swiss 2 km grid), with their operational settings, and
the interpolation of static fields (25 m and 50 m grids, lazy with dask). The data is synthetic, with the structure,
names, CRS attributes, sizes, dtypes and NaN patterns of the operational data, or the real data with ``--data-dir``.

Each case runs in fresh processes of this script: the first call has cold caches (as in the operational cron jobs),
the next ones warm caches (several interpolations within a job). The results are for comparing two versions of
grid-process on the same machine, not absolute numbers. On a shared machine, the runtimes vary by up to about 30 %
with the load (``load_average_time`` is recorded per case): rerun the cases that differ.

On the operational data, the synthetic cases take the same runtime (within about 20 %) and memory (within 5 %),
except the static fields: reading and decompressing the zarr stores dominates their runtime (e.g. 64 s for the
first call of the horizon, 18 s for the next ones), whereas the synthetic chunks cost little. The synthetic static
cases measure what grid-process adds (dask graph, slicing, indexing): use ``--data-dir`` for their absolute cost.

Measured per case:

- ``first_s``, ``repeat_s``: runtime of the first call and of the next ones (``--repeat``), outputs computed;
- ``rss_first_gb``, ``rss_repeat_gb``: increase of the peak resident memory during the first and the second call
  (Linux only, includes the memory allocated outside of Python, e.g. KDTree);
- ``traced_first_gb``, ``traced_repeat_gb``: peak of the memory allocated through Python (numpy, dask) during the
  first and the second call, measured with tracemalloc in a separate process (unless ``--no-memory``);
- per output variable: a checksum (identical outputs), the NaN count, mean and standard deviation.

Examples
--------
List the cases::

    python benchmarks/operational_cases.py list

Compare a branch with main::

    git worktree add ../grid-process-main main
    python benchmarks/operational_cases.py run --grid-process ../grid-process-main --out /tmp/bench/main
    python benchmarks/operational_cases.py run --out /tmp/bench/branch
    python benchmarks/operational_cases.py compare /tmp/bench/main /tmp/bench/branch

Select cases with shell patterns (``--case 'grid_icon_ch1*' static_horizon_poi``) and keep the outputs to quantify
their differences (``--save-outputs``).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
import warnings
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dask.array
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

N_POIS = 5622
LV95 = "EPSG:2056"
ROTATED = {
    "earth_radius": 6371229.0,
    "grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": 43.0,
    "grid_north_pole_longitude": 190.0,
    "north_pole_grid_longitude": 0.0,
}
LV95_ICON = {
    "epsg_code": LV95,
    "false_easting": 2600000.0,
    "false_northing": 1200000.0,
    "semi_major_axis": 6377397.0,
    "semi_minor_axis": 6356079.0,
    "grid_mapping_name": "Oblique Mercator (LV95)",
    "inverse_flattening": 299.1528015136719,
    "latitude_of_projection_center": 46.9524040222168,
    "longitude_of_projection_center": 7.4395833015441895,
}
LV95_INCA = {
    "epsg_code": LV95,
    "long_name": "Swiss coordinate system",
    "false_easting": 2600000.0,
    "false_northing": 1200000.0,
    "semi_major_axis": 6377397.155,
    "semi_minor_axis": 6356078.962822,
    "grid_mapping_name": "Oblique Mercator (CH1903+ / LV95)",
    "inverse_flattening": 299.1528128,
    "latitude_of_projection_center": 46.95240555555556,
    "longitude_of_projection_center": 7.439583333333333,
}
# lakes as ellipses (center E, center N, semi-axis along E, semi-axis along N) [m, LV95]: about 2 % of the points
# of the ICON-CH2-EPS grid, as the lake mask land_area_fraction < 0.5
LAKES = (
    (2530000, 1140000, 30000, 6000),  # Geneva
    (2735000, 1280000, 26000, 7000),  # Constance
    (2555000, 1197000, 17000, 3500),  # Neuchâtel
    (2705000, 1095000, 5000, 26000),  # Maggiore
    (2735000, 1075000, 4000, 20000),  # Como
    (2835000, 1020000, 7000, 22000),  # Garda
    (2690000, 1235000, 3500, 14000),  # Zurich
)
STATIC_ATTRS = {"crs": LV95}
HORIZON_ATTRS = {"fill_val": -127, "offset": 40, "scale_fact": 0.4, "units": "degree"}


@dataclass(frozen=True)
class Grid:  # pylint: disable=too-many-instance-attributes
    """
    Structured grid of the grid database: regular 1D coordinates ``x``, ``y`` in the CRS of ``grid_mapping``, and
    the sizes of the other dimensions of the fields, in order before (y, x).
    """

    x0: float
    dx: float
    nx: int
    y0: float
    dy: float
    ny: int
    grid_mapping: dict[str, Any]
    sizes: dict[str, int]


GRIDS = {
    "ifs": Grid(
        0.1, 0.1, 168, 43.6, 0.1, 59, {"grid_mapping_name": "latitude_longitude"},
        {"forecast_reference_time": 1, "lead_time": 145, "realization": 51},
    ),
    "icon_ch1": Grid(
        -3.15, 0.01, 376, -1.41, 0.01, 252, ROTATED,
        {"forecast_reference_time": 1, "lead_time": 34, "realization": 11},
    ),
    "icon_ch2": Grid(
        -3.14, 0.02, 188, -1.42, 0.02, 127, ROTATED,
        {"forecast_reference_time": 1, "lead_time": 121, "realization": 21},
    ),
    "icon_ch2_lv95": Grid(
        2256000, 2000, 355, 841000, 2000, 320, LV95_ICON,
        {"forecast_reference_time": 1, "lead_time": 121, "realization": 21},
    ),
    "inca": Grid(
        2255500, 1000, 710, 840500, 1000, 640, LV95_INCA, {"forecast_reference_time": 18, "lead_time": 1},
    ),
}  # fmt: skip

# source name: (file name in --data-dir, grid, variable)
GRID_SOURCES = {
    "ifs": ("ecmwf_ifs_gridefix.nc", "ifs", "air_temperature"),
    "icon_ch1": ("icon_ch1_gridefix.nc", "icon_ch1", "surface_diffuse_downwelling_shortwave_flux_in_air"),
    "icon_ch1_snow": ("icon_ch1_snowlimit_nans_gridefix.nc", "icon_ch1", "snow_limit"),
    "icon_ch2": ("icon_ch2_gridefix.nc", "icon_ch2", "surface_diffuse_downwelling_shortwave_flux_in_air"),
    "icon_ch2_lake": ("icon_ch2_temperature_lake_masked_gridefix.nc", "icon_ch2", "air_temperature"),
    "icon_ch2_lv95": ("icon_ch2_lv95_gridefix.nc", "icon_ch2_lv95", "air_temperature"),
    "inca": ("inca_gridefix.nc", "inca", "cloud_area_fraction_in_low_troposphere"),
}
OTHER_FILES = {
    "icon_ch1_native": "icon_ch1_native.nc",
    "pois": "pois.geojson",
    "dst": "seamless_2km.nc",
    "horizon": "swiss_25/horizon_HORIZON.zarr",
    "topo50": "radar_50/topo_TPI_500M.zarr",
    "svf50": "radar_50/horizon_SVF.zarr",
    "dem50": "radar_50/DEM.zarr",
}


# ----------------------------------------------------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------------------------------------------------


def orography(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """
    Synthetic surface altitude [m], between 0 and 4 500 m, at LV95 coordinates [m]: an alpine ridge from south-west
    to east with valleys, and hills elsewhere. The same function gives the altitude of all the grids and POIs.
    """

    e = (np.asarray(east, dtype=np.float64) - 2600000.0) / 1000.0
    n = (np.asarray(north, dtype=np.float64) - 1200000.0) / 1000.0
    ridge = 3300.0 * np.exp(-(((n + 75.0 - 0.12 * e) / 45.0) ** 2)) * (0.7 + 0.3 * np.sin(e / 11.0) * np.sin(n / 7.0))
    return np.clip(450.0 + ridge + 180.0 * np.sin(e / 3.1) * np.cos(n / 2.3), 0.0, 4500.0)


def _rng(name: str) -> np.random.Generator:
    """Random generator seeded by ``name``: the same data whatever the selected cases."""

    return np.random.default_rng(zlib.crc32(name.encode()))


def _scaled(size: int, scale: float, minimum: int) -> int:
    return size if scale == 1 else max(minimum, round(size * scale))


def _axis(start: float, step: float, size: int, scale: float, minimum: int = 16) -> np.ndarray:
    """Regular axis covering the same extent with fewer points if ``scale`` < 1."""

    new_size = _scaled(size, scale, minimum)
    new_step = step if new_size == size else step * (size - 1) / (new_size - 1)
    return start + new_step * np.arange(new_size)


def _grid_crs(grid_mapping: dict[str, Any]) -> CRS:
    return CRS(grid_mapping["epsg_code"]) if "epsg_code" in grid_mapping else CRS.from_cf(grid_mapping)


def _lake_mask(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    mask = np.zeros(np.shape(east), dtype=bool)
    for e0, n0, a, b in LAKES:
        mask |= ((east - e0) / a) ** 2 + ((north - n0) / b) ** 2 <= 1
    return mask


def _field(variable: str, shape: tuple[int, ...], alt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    float32 values of ``variable`` of shape ``shape`` (spatial dimensions last, lead time second), allocated once
    (filled in place) to keep the memory of the data building at the size of the data.
    """

    if variable == "cloud_area_fraction_in_low_troposphere":
        return rng.random(shape, dtype=np.float32)
    values = rng.standard_normal(shape, dtype=np.float32)
    if variable == "air_temperature":
        values *= 1.5
        values += (288.0 - 0.0065 * alt).astype(np.float32)
    elif variable == "surface_diffuse_downwelling_shortwave_flux_in_air":
        values *= 30.0
        values += 150.0
        np.maximum(values, 0.0, out=values)
        values[:, 0] = np.nan  # as the operational data: no radiation at the first lead time
    elif variable == "snow_limit":
        # NaN where the snow limit is below the orography: levels between the 62nd and 76th percentiles of the
        # altitude, i.e. about 31 % of NaN values and 24 % of points NaN for all lead times and members
        low, high = np.percentile(alt, (62, 76))
        levels = rng.uniform(low, high, shape[:-2] + (1, 1)).astype(np.float32)
        values *= 20.0
        values += levels
        np.minimum(values, high, out=values)
        values[values < alt] = np.nan
    else:
        raise ValueError(f"No synthetic values for {variable!r}")
    return values


def grid_database(source: str, scale: float = 1.0) -> xr.Dataset:  # pylint: disable=too-many-locals
    """
    Synthetic dataset of the grid database ``source`` (see ``GRID_SOURCES``): same dimensions, coordinates,
    attributes, sizes (reduced if ``scale`` < 1) and NaN patterns as the operational data.
    """

    _, grid_name, variable = GRID_SOURCES[source]
    grid = GRIDS[grid_name]
    rng = _rng(source)
    x = _axis(grid.x0, grid.dx, grid.nx, scale).astype(np.float32)
    y = _axis(grid.y0, grid.dy, grid.ny, scale).astype(np.float32)
    xx, yy = np.meshgrid(x.astype(np.float64), y.astype(np.float64))
    crs = _grid_crs(grid.grid_mapping)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    east, north = Transformer.from_crs(crs, LV95, always_xy=True).transform(xx, yy)
    alt = orography(east, north).astype(np.float32)

    sizes = {dim: _scaled(size, scale, 2 if dim == "lead_time" and size > 1 else 1) for dim, size in grid.sizes.items()}
    values = _field(variable, tuple(sizes.values()) + alt.shape, alt, rng)
    if source == "icon_ch2_lake":
        values[..., _lake_mask(east, north)] = np.nan

    reftime = np.datetime64("2026-11-19T00:00", "ns") + np.arange(sizes["forecast_reference_time"]) * np.timedelta64(
        10, "m"
    )
    leadtime = np.arange(sizes["lead_time"]) * np.timedelta64(1, "h").astype("m8[ns]")
    coords: dict[str, Any] = {
        "x": x,
        "y": y,
        "lead_time": leadtime,
        "forecast_reference_time": reftime,
        "longitude": (("y", "x"), lon.astype(np.float32)),
        "latitude": (("y", "x"), lat.astype(np.float32)),
        "time": (("forecast_reference_time", "lead_time"), reftime[:, None] + leadtime[None, :]),
        "surface_altitude": (("y", "x"), alt),
    }
    if "realization" in sizes:
        coords["realization"] = np.arange(sizes["realization"], dtype=np.float64)
    return xr.Dataset(
        {variable: (tuple(sizes) + ("y", "x"), values)},
        coords=coords,
        attrs={"source": source, "grid_mapping": json.dumps(grid.grid_mapping)},
    )


def icon_native(scale: float = 1.0) -> xr.Dataset:  # pylint: disable=too-many-locals
    """
    Synthetic native ICON-CH1 forecast (control member): 1 147 980 points (dimension ``values``) on a jittered
    lattice covering the native domain, with the 2 m temperature at 34 lead times and the surface altitude ``HSURF``.
    """

    rng = _rng("icon_ch1_native")
    n_points = _scaled(1147980, scale**2, 256)
    x_min, x_max, y_min, y_max = -6.96, 4.94, -4.56, 3.50  # extent of the native domain in rotated coordinates
    nx = round(np.sqrt(n_points * (x_max - x_min) / (y_max - y_min)))
    ny = -(-n_points // nx)
    step = (x_max - x_min) / nx
    yy, xx = np.divmod(np.arange(n_points), nx)
    rlon = x_min + (xx + 0.5 + rng.uniform(-0.3, 0.3, n_points)) * step
    rlat = y_min + (yy + 0.5 + rng.uniform(-0.3, 0.3, n_points)) * (y_max - y_min) / ny
    lon, lat = Transformer.from_crs(CRS.from_cf(ROTATED), "EPSG:4326", always_xy=True).transform(rlon, rlat)
    east, north = Transformer.from_crs("EPSG:4326", LV95, always_xy=True).transform(lon, lat)
    alt = orography(east, north).astype(np.float32)
    n_steps = _scaled(34, scale, 2)
    values = _field("air_temperature", (1, n_steps, n_points), alt, rng)
    return xr.Dataset(
        {"T_2M": (("forecast_reference_time", "step", "values"), values)},
        coords={
            "forecast_reference_time": [np.datetime64("2026-11-19T00:00", "ns")],
            "step": np.arange(n_steps) * np.timedelta64(1, "h").astype("m8[ns]"),
            "latitude": ("values", lat),
            "longitude": ("values", lon),
            "HSURF": ("values", alt),
        },
    )


def pois(scale: float = 1.0) -> gpd.GeoDataFrame:
    """Synthetic POIs: 5 622 points in the bounds of Switzerland, with their altitude (``height_masl``)."""

    rng = _rng("pois")
    n_pois = _scaled(N_POIS, scale, 20)
    east = rng.uniform(2485000, 2834000, n_pois)
    north = rng.uniform(1075000, 1296000, n_pois)
    height = orography(east, north) + rng.normal(0, 100, n_pois)
    lon, lat = Transformer.from_crs(LV95, "EPSG:4326", always_xy=True).transform(east, north)
    return gpd.GeoDataFrame(
        {"name": [f"poi_{i:05d}" for i in range(n_pois)], "height_masl": height},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )


def dst_grid(scale: float = 1.0) -> xr.Dataset:
    """Synthetic 2 km LV95 destination grid of the seamless weather provider, with its surface altitude."""

    x = _axis(2471000, 2000, 190, scale)
    y = _axis(1061000, 2000, 125, scale)
    alt = orography(*np.meshgrid(x, y))
    return xr.Dataset(coords={"x": x, "y": y, "surface_altitude": (("y", "x"), alt)}, attrs={"crs": LV95})


def _static_profiles(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Altitude profiles along x and y [m] of the static fields, whose sum is the altitude: separable, so that a chunk
    costs a broadcast sum (about the cost of reading a chunk of a zarr store), not an evaluation of the orography.
    """

    return 1500.0 + 700.0 * np.sin(x / 37000.0), 600.0 * np.cos(y / 23000.0)


def _lazy_static(
    x: np.ndarray, y: np.ndarray, chunks: tuple[int, int], func: Callable[[np.ndarray], np.ndarray], dtype: Any
) -> dask.array.Array:
    """Lazy (dask) array of shape (y, x) computed per chunk by ``func`` from the altitude (see _static_profiles)."""

    alt_x, alt_y = _static_profiles(x, y)

    def block(block_info: dict[Any, Any] | None = None) -> np.ndarray:
        assert block_info is not None
        (y_start, y_end), (x_start, x_end) = block_info[None]["array-location"][:2]
        alt = alt_y[y_start:y_end, None].astype(dtype) + alt_x[None, x_start:x_end].astype(dtype)
        return func(alt).astype(dtype, copy=False)

    return dask.array.map_blocks(
        block, chunks=dask.array.core.normalize_chunks(chunks, (len(y), len(x))), dtype=dtype, meta=np.array((), dtype)
    )


def static_50m(scale: float = 1.0) -> xr.Dataset:
    """
    Synthetic static fields on the 50 m LV95 grid of the Swiss radar domain, lazy (dask) with the chunks of the
    zarr stores: DEM, topographic position index and sky view factor.
    """

    x = _axis(2255025, 50, 14200, scale)
    y = _axis(840025, 50, 12800, scale)
    chunks = (_scaled(6400, scale, 8), _scaled(4800, scale, 8))
    data = {
        "dem": _lazy_static(x, y, chunks, lambda alt: alt, np.float32),
        "tpi_500m": _lazy_static(x, y, chunks, lambda alt: alt * np.float32(0.02) - 30, np.float32),
        "svf": _lazy_static(x, y, chunks, lambda alt: 1 - alt * np.float32(1e-4), np.float32),
    }
    return xr.Dataset({name: (("y", "x"), array) for name, array in data.items()}, {"x": x, "y": y}, STATIC_ATTRS)


def horizon(scale: float = 1.0) -> xr.Dataset:
    """
    Synthetic horizon on the 25 m LV95 grid of Switzerland, lazy (dask) with the chunks of the zarr store: int8
    angles per azimuth (0 to 359 degrees), encoded as the store (``fill_val``, ``scale_fact``, ``offset``).
    """

    x = _axis(2460000, 25, 15601, scale)
    y = _axis(1060000, 25, 10121, scale)
    azimuth = np.arange(360, dtype=np.float64)
    alt_x, alt_y = _static_profiles(x, y)  # encoded angles from -60 to 30, plus up to 20 depending on the azimuth
    angle_x, angle_y = (alt_x / 40.0 - 60.0).astype(np.int8), (alt_y / 40.0).astype(np.int8)
    shift = (np.cos(np.deg2rad(azimuth)) * 20.0).astype(np.int8)

    def block(block_info: dict[Any, Any] | None = None) -> np.ndarray:
        assert block_info is not None
        (y_start, y_end), (x_start, x_end), (a_start, a_end) = block_info[None]["array-location"]
        angle = angle_y[y_start:y_end, None] + angle_x[None, x_start:x_end]
        return angle[:, :, None] + shift[None, None, a_start:a_end]

    chunks = (_scaled(500, scale, 8), _scaled(500, scale, 8), 45)
    data = dask.array.map_blocks(
        block,
        chunks=dask.array.core.normalize_chunks(chunks, (len(y), len(x), len(azimuth))),
        dtype=np.int8,
        meta=np.array((), np.int8),
    )
    return xr.Dataset(
        {"horizon": (("y", "x", "azimuth"), data, HORIZON_ATTRS)},
        {"x": x, "y": y, "azimuth": azimuth},
        STATIC_ATTRS,
    )


# ----------------------------------------------------------------------------------------------------------------
# inputs: synthetic or real
# ----------------------------------------------------------------------------------------------------------------


def load_input(name: str, scale: float = 1.0, data_dir: Path | None = None) -> tuple[Any, bool]:
    """
    Input ``name`` (a key of ``GRID_SOURCES`` or ``OTHER_FILES``), from ``data_dir`` if its file is there (real
    data), otherwise synthetic. Returns the input and whether it is real.
    """

    file_name = GRID_SOURCES[name][0] if name in GRID_SOURCES else OTHER_FILES[name]
    path = None if data_dir is None else data_dir / file_name
    if path is None or not path.exists():
        return _synthetic(name, scale), False
    if name in GRID_SOURCES:
        variable = GRID_SOURCES[name][2]
        with xr.open_dataset(path) as ds:
            return ds[[variable]].load(), True
    if name == "icon_ch1_native":
        with xr.open_dataset(path) as ds:
            return ds[["T_2M"]].load(), True
    if name == "pois":
        return gpd.read_file(path), True
    if name == "dst":
        with xr.open_dataset(path) as ds:
            coords = {"x": ds.x.values, "y": ds.y.values, "surface_altitude": (("y", "x"), ds.dem.values)}
            return xr.Dataset(coords=coords, attrs={"crs": LV95}), True
    return xr.open_zarr(path), True


def _synthetic(name: str, scale: float) -> Any:
    if name in GRID_SOURCES:
        return grid_database(name, scale)
    builders: dict[str, Callable[[float], Any]] = {
        "icon_ch1_native": icon_native,
        "pois": pois,
        "dst": dst_grid,
        "horizon": horizon,
        "static50": static_50m,
    }
    return builders[name](scale)


def _static50(scale: float, data_dir: Path | None) -> tuple[xr.Dataset, bool]:
    """The 50 m static fields: real stores merged if all are in ``data_dir``, otherwise synthetic."""

    if data_dir is not None and all((data_dir / OTHER_FILES[name]).exists() for name in ("topo50", "svf50", "dem50")):
        return xr.merge([xr.open_zarr(data_dir / OTHER_FILES[name]) for name in ("dem50", "topo50", "svf50")]), True
    return static_50m(scale), False


# ----------------------------------------------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """
    Operational case: ``call(gri, source, target)`` with the module ``grid_process.grid_interp``, the source data
    and the target (POIs or destination grid), returning the computed output.
    """

    name: str
    source: str
    target: str
    description: str
    call: Callable[[Any, Any, Any], xr.Dataset]


def _to_pois(method: str, **kwargs: Any) -> Callable[[Any, Any, Any], xr.Dataset]:
    """data4web pipelines: interpolation to the POIs."""

    def call(gri: Any, source: xr.Dataset, target: gpd.GeoDataFrame) -> xr.Dataset:
        return gri.interp2poi(source, target, method, try_reg=False, **kwargs).load()

    return call


def _to_grid(method: str, **kwargs: Any) -> Callable[[Any, Any, Any], xr.Dataset]:
    """Seamless weather provider: slice (and average) to the destination grid, then interpolation."""

    def call(gri: Any, source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
        dims = ("y", "x")
        sliced = gri.slice_coarsen(source, target, method, spatial_dims=dims, dst_spatial_dims=dims, **kwargs)
        return gri.interp2grid(sliced, target, method, try_reg=False, **kwargs).load()

    return call


def _native_to_grid(gri: Any, source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    return gri.interp2grid(source, target, "idw", try_reg=False, k=4, weight_power=0.5).load()


def _horizon_to_pois(gri: Any, source: xr.Dataset, target: gpd.GeoDataFrame) -> xr.Dataset:
    """data4web process_horizon"""

    out = gri.interp2poi(source.isel(azimuth=slice(45, -45)), target, method="nearest")
    return out.astype(np.float32, copy=False).load()


def _static_to_pois(gri: Any, source: xr.Dataset, target: gpd.GeoDataFrame) -> xr.Dataset:
    """data4web mosmix topographic descriptors"""

    return gri.interp2poi(source, target, method="linear").load().astype(np.float32, copy=False)


def _dem_to_grid(gri: Any, source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Seamless weather provider: elevation of the destination grid"""

    return gri.interp2grid(source[["dem"]], target.drop_vars("surface_altitude"), method="linear").load()


SNOWLIM_CH1 = {"k": 100, "d_max": 40000}
TEMP = {"k": 50, "weight_power": 0.75, "z_fact": 15}
CASES = [
    Case("poi_icon_ch1_nearest_alt", "icon_ch1", "pois", "", _to_pois("nearest_alt", r_search=1.8)),
    Case("poi_icon_ch1_idw_k49", "icon_ch1", "pois", "", _to_pois("idw", k=49, weight_power=0.125)),
    Case("poi_icon_ch1_nearest", "icon_ch1", "pois", "", _to_pois("nearest")),
    Case("poi_icon_ch1_snowlim", "icon_ch1_snow", "pois", "snow limit with NaN", _to_pois("snowlim", **SNOWLIM_CH1)),
    Case("poi_icon_ch2_idw_k4", "icon_ch2", "pois", "", _to_pois("idw", k=4, weight_power=0.5)),
    Case("poi_icon_ch2_lake_temp", "icon_ch2_lake", "pois", "lakes masked", _to_pois("temp", **TEMP)),
    Case("poi_ifs_nearest_alt", "ifs", "pois", "", _to_pois("nearest_alt", r_search=2)),
    Case("poi_ifs_temp", "ifs", "pois", "", _to_pois("temp", **TEMP)),
    Case("poi_inca_idw_k38", "inca", "pois", "", _to_pois("idw", k=38, weight_power=0.125)),
    Case("poi_inca_temp", "inca", "pois", "", _to_pois("temp", k=36, weight_power=0.5, z_fact=4)),
    Case("poi_icon_ch1_native_idw", "icon_ch1_native", "pois", "", _to_pois("idw", k=4, weight_power=0.5)),
    Case("poi_icon_ch1_native_upscaled", "icon_ch1_native", "pois", "LPI (planned)", _to_pois("upscaled", k=289)),
    Case("grid_icon_ch1_temp_fixgrad", "icon_ch1", "dst", "", _to_grid("temp_fixgrad", r_search=1.7)),
    Case("grid_icon_ch1_nearest_alt", "icon_ch1", "dst", "", _to_grid("nearest_alt", r_search=1.7)),
    Case("grid_icon_ch1_idw", "icon_ch1", "dst", "", _to_grid("idw", k=4, weight_power=0.5)),
    Case(
        "grid_icon_ch1_snow_nearest_alt", "icon_ch1_snow", "dst", "snow limit with NaN (future)",
        _to_grid("nearest_alt", r_search=1.7),
    ),
    Case("grid_icon_ch2_nearest_alt", "icon_ch2", "dst", "", _to_grid("nearest_alt", r_search=1.7)),
    Case(
        "grid_icon_ch2_lake_temp_fixgrad", "icon_ch2_lake", "dst", "lakes masked",
        _to_grid("temp_fixgrad", r_search=1.7),
    ),
    Case("grid_icon_ch2_lv95_temp_fixgrad", "icon_ch2_lv95", "dst", "", _to_grid("temp_fixgrad", r_search=1.7)),
    Case("grid_icon_ch2_lv95_nearest_alt", "icon_ch2_lv95", "dst", "", _to_grid("nearest_alt", r_search=1.7)),
    Case("grid_ifs_temp_fixgrad", "ifs", "dst", "", _to_grid("temp_fixgrad", r_search=2)),
    Case("grid_ifs_idw", "ifs", "dst", "", _to_grid("idw", k=4, weight_power=1)),
    Case("grid_inca_idw", "inca", "dst", "", _to_grid("idw", k=4, weight_power=2)),
    Case("grid_inca_nearest_alt", "inca", "dst", "", _to_grid("nearest_alt", r_search=1.2)),
    Case("grid_icon_ch1_native_idw", "icon_ch1_native", "dst", "interp2grid only", _native_to_grid),
    Case("static_horizon_poi", "horizon", "pois", "data4web process_horizon, regular path", _horizon_to_pois),
    Case("static_topo_poi_linear", "static50", "pois", "mosmix descriptors, regular path", _static_to_pois),
    Case("static_dem_grid_linear", "static50", "dst", "seamless elevation, regular path", _dem_to_grid),
]  # fmt: skip
CASES_BY_NAME = {case.name: case for case in CASES}


def case_inputs(case: Case, scale: float = 1.0, data_dir: Path | None = None) -> tuple[Any, Any, dict[str, bool]]:
    """Source and target of ``case``, and whether each is real data."""

    if case.source == "static50":
        source, source_real = _static50(scale, data_dir)
    else:
        source, source_real = load_input(case.source, scale, data_dir)
    target, target_real = load_input(case.target, scale, data_dir)
    return source, target, {case.source: source_real, case.target: target_real}


# ----------------------------------------------------------------------------------------------------------------
# measurements (worker processes)
# ----------------------------------------------------------------------------------------------------------------


def _proc_status(key: str) -> int | None:
    """Value [bytes] of ``key`` (e.g. VmRSS, VmHWM) in /proc/self/status, None if not available (not Linux)."""

    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _reset_peak_rss() -> bool:
    """Reset the peak resident memory (VmHWM) of the process (Linux 4.0 or later)."""

    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as clear_refs:
            clear_refs.write("5")
        return True
    except OSError:
        return False


def _timed(call: Callable[[], xr.Dataset]) -> tuple[xr.Dataset, float, float | None]:
    """Output, runtime [s] and increase of the peak resident memory [GB] (None if not available) of ``call``."""

    rss = _proc_status("VmRSS") if _reset_peak_rss() else None
    start = time.perf_counter()
    out = call()
    runtime = time.perf_counter() - start
    peak = _proc_status("VmHWM")
    return out, runtime, None if rss is None or peak is None else (peak - rss) / 1e9


def _traced(call: Callable[[], xr.Dataset]) -> float:
    """Peak [GB] of the memory allocated through Python during ``call``, tracemalloc being started."""

    tracemalloc.reset_peak()
    current = tracemalloc.get_traced_memory()[0]
    call()
    return (tracemalloc.get_traced_memory()[1] - current) / 1e9


def _summary(out: xr.Dataset) -> dict[str, dict[str, Any]]:
    summary = {}
    for name, var in out.data_vars.items():
        values = np.ascontiguousarray(var.values)
        digest = hashlib.sha256(f"{values.dtype.str}{values.shape}{var.dims}".encode())
        digest.update(values.tobytes())
        floats = values.astype(np.float64)
        summary[str(name)] = {
            "dims": dict(var.sizes),
            "dtype": values.dtype.str,
            "sha256": digest.hexdigest(),
            "nan": int(np.isnan(floats).sum()),
            "mean": float(np.nanmean(floats)) if np.isfinite(floats).any() else None,
            "std": float(np.nanstd(floats)) if np.isfinite(floats).any() else None,
        }
    return summary


def _environment(grid_process: Any) -> dict[str, Any]:
    import scipy  # pylint: disable=import-outside-toplevel

    root = Path(grid_process.__file__).resolve().parents[1]
    git: dict[str, Any] = {}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
        git = {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "grid_process": str(root),
        "git": git,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "xarray": xr.__version__,
            "dask": dask.__version__,
            "pandas": pd.__version__,
        },
        "host": platform.node(),
        "cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
    }


def worker(args: argparse.Namespace) -> None:
    """Run one case in this process (``mode`` time or memory) and write its results as JSON."""

    warnings.simplefilter("ignore")
    import grid_process  # pylint: disable=import-outside-toplevel
    import grid_process.grid_interp as gri  # pylint: disable=import-outside-toplevel

    if args.grid_process and not Path(grid_process.__file__).resolve().is_relative_to(args.grid_process.resolve()):
        raise RuntimeError(f"grid_process imported from {grid_process.__file__}, not from {args.grid_process}")
    case = CASES_BY_NAME[args.case]
    source, target, real = case_inputs(case, args.scale, args.data_dir)

    def call() -> xr.Dataset:
        return case.call(gri, source, target)

    result: dict[str, Any] = {"case": case.name, "real_data": real}
    if hasattr(os, "getloadavg"):
        result[f"load_average_{args.mode}"] = os.getloadavg()[0]
    if args.mode == "time":
        out, result["first_s"], result["rss_first_gb"] = _timed(call)
        repeats = [_timed(call) for _ in range(args.repeat)]
        result["repeat_s"] = [runtime for _, runtime, _ in repeats]
        result["rss_repeat_gb"] = repeats[0][2] if repeats else None
        result["outputs"] = _summary(out)
        result["environment"] = _environment(grid_process)
        if args.save_outputs:
            out.attrs = {}
            for var in out.variables.values():
                var.attrs = {}
            out.to_netcdf(args.out / "outputs" / f"{case.name}.nc")
    else:
        tracemalloc.start()
        result["traced_first_gb"] = _traced(call)
        result["traced_repeat_gb"] = _traced(call)
        tracemalloc.stop()
    (args.out / "cases" / f"{case.name}.{args.mode}.json").write_text(json.dumps(result, indent=1))


# ----------------------------------------------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Run the selected cases, each in fresh worker processes, and write ``results.json`` in ``--out``."""

    cases = [case for case in CASES if any(fnmatch.fnmatch(case.name, pattern) for pattern in args.case)]
    if not cases:
        raise SystemExit(f"No case matches {args.case}, see the list command")
    for sub_dir in ("cases", "outputs"):
        (args.out / sub_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.grid_process:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(args.grid_process.resolve()), env.get("PYTHONPATH"))))
    common = ["--out", str(args.out), "--scale", str(args.scale), "--repeat", str(args.repeat)]
    common += ["--grid-process", str(args.grid_process)] if args.grid_process else []
    common += ["--data-dir", str(args.data_dir)] if args.data_dir else []
    common += ["--save-outputs"] if args.save_outputs else []

    results: dict[str, Any] = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cases": {}}
    for case in cases:
        result: dict[str, Any] = {}
        for mode in ("time",) if args.no_memory else ("time", "memory"):
            path = args.out / "cases" / f"{case.name}.{mode}.json"
            path.unlink(missing_ok=True)
            command = [sys.executable, __file__, "_worker", "--case", case.name, "--mode", mode] + common
            process = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
            if path.exists():  # also if the interpreter fails at exit after writing the results
                result.update(json.loads(path.read_text()))
            else:
                result.setdefault("errors", []).append(process.stderr.strip().splitlines()[-1:])
        results.setdefault("environment", result.pop("environment", None))
        results["cases"][case.name] = result
        print(_case_line(case.name, result), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))


def _case_line(name: str, result: dict[str, Any]) -> str:
    if "first_s" not in result:
        return f"{name:34s} FAILED {result.get('errors')}"
    repeat = min(result["repeat_s"]) if result["repeat_s"] else float("nan")
    memory = f"{result['traced_first_gb']:6.2f} GB" if "traced_first_gb" in result else ""
    real = " (real data)" if any(result["real_data"].values()) else ""
    return f"{name:34s} first {result['first_s']:7.2f} s  repeat {repeat:7.2f} s  traced peak {memory}{real}"


def compare(args: argparse.Namespace) -> None:
    """Print the runtimes, peaks and output differences of the cases of two runs (``--out`` directories)."""

    runs = [json.loads((path / "results.json").read_text()) for path in (args.a, args.b)]
    for label, path, result in zip("AB", (args.a, args.b), runs):
        env = result.get("environment") or {}
        loads = [case.get("load_average_time", 0.0) for case in result["cases"].values()]
        print(f"{label}: {path}, grid_process {env.get('grid_process')} {env.get('git')}, {env.get('cpus')} CPUs,")
        print(f"   load average at the start of the cases up to {max(loads, default=0.0):.1f}")
    header = f"{'case':34s} {'first [s]':>17s} {'repeat [s]':>17s} {'traced peak [GB]':>17s} {'rss peak [GB]':>17s}"
    print(f"{header}  outputs")
    for name, res_a in runs[0]["cases"].items():
        res_b = runs[1]["cases"].get(name)
        if res_b is None or "first_s" not in res_a or "first_s" not in res_b:
            continue
        columns = [
            _ratio(res_a["first_s"], res_b["first_s"]),
            _ratio(min(res_a["repeat_s"], default=None), min(res_b["repeat_s"], default=None)),
            _ratio(res_a.get("traced_first_gb"), res_b.get("traced_first_gb")),
            _ratio(res_a.get("rss_first_gb"), res_b.get("rss_first_gb")),
        ]
        print(f"{name:34s} {' '.join(columns)}  {_output_diff(name, res_a, res_b, args.a, args.b)}")


def _ratio(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return f"{'-':>17s}"
    return f"{a:6.2f} {b:6.2f} {b / a if a else float('nan'):3.2f}x"


def _output_diff(  # pylint: disable=too-many-locals
    name: str, res_a: dict[str, Any], res_b: dict[str, Any], dir_a: Path, dir_b: Path
) -> str:
    out_a, out_b = res_a["outputs"], res_b["outputs"]
    if out_a.keys() != out_b.keys():
        return f"different variables {sorted(out_a)} {sorted(out_b)}"
    if all(out_a[var]["sha256"] == out_b[var]["sha256"] for var in out_a):
        return "identical"
    paths = [path / "outputs" / f"{name}.nc" for path in (dir_a, dir_b)]
    if not all(path.exists() for path in paths):
        diffs = [f"{var}: mean {out_a[var]['mean']} {out_b[var]['mean']}, NaN {out_a[var]['nan']} {out_b[var]['nan']}"
                 for var in out_a if out_a[var]["sha256"] != out_b[var]["sha256"]]  # fmt: skip
        return "different (--save-outputs for details) " + "; ".join(diffs)
    with xr.open_dataset(paths[0]) as ds_a, xr.open_dataset(paths[1]) as ds_b:
        diffs = []
        for var in out_a:
            a, b = ds_a[var].values.astype(np.float64), ds_b[var].values.astype(np.float64)
            if a.shape != b.shape:
                diffs.append(f"{var}: shapes {a.shape} {b.shape}")
                continue
            nan_diff = int((np.isnan(a) != np.isnan(b)).sum())
            delta = np.abs(a - b)
            delta_max = float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0
            rmse = float(np.sqrt(np.nanmean(delta**2))) if np.isfinite(delta).any() else 0.0
            diffs.append(f"{var}: max {delta_max:.3g}, RMSE {rmse:.3g}, NaN differ {nan_diff}")
        return "; ".join(diffs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the cases")

    run_parser = commands.add_parser("run", help="run the cases")
    worker_parser = commands.add_parser("_worker")
    for sub_parser in (run_parser, worker_parser):
        sub_parser.add_argument("--out", type=Path, required=True, help="output directory (outside the repository)")
        sub_parser.add_argument("--repeat", type=int, default=3, help="number of calls after the first one")
        sub_parser.add_argument("--grid-process", type=Path, help="grid-process checkout to benchmark")
        sub_parser.add_argument("--data-dir", type=Path, help="directory with the real data (see OTHER_FILES)")
        sub_parser.add_argument("--scale", type=float, default=1.0, help="scale of the sizes (< 1: smoke tests)")
        sub_parser.add_argument("--save-outputs", action="store_true", help="keep the outputs to compare them")
    run_parser.add_argument("--case", nargs="+", default=["*"], help="shell patterns of the case names")
    run_parser.add_argument("--no-memory", action="store_true", help="skip the tracemalloc process")
    worker_parser.add_argument("--case", required=True)
    worker_parser.add_argument("--mode", choices=("time", "memory"), required=True)

    compare_parser = commands.add_parser("compare", help="compare two runs")
    compare_parser.add_argument("a", type=Path, help="output directory of the reference run")
    compare_parser.add_argument("b", type=Path, help="output directory of the run to compare")

    args = parser.parse_args(argv)
    if args.command == "list":
        for case in CASES:
            print(f"{case.name:34s} {case.source:16s} -> {case.target:5s} {case.description}")
    elif args.command == "run":
        run(args)
    elif args.command == "_worker":
        worker(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
