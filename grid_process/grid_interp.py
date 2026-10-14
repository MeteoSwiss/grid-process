# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=too-many-lines

"""
Module for spatial interpolation of gridded datasets
"""

import logging
from functools import partial
from typing import Any, Literal, overload
from collections.abc import Callable
from math import isclose, prod, ceil

import numpy as np
from numpy.typing import NDArray
import xarray as xr
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from pyproj import CRS

from grid_process import CONFIG
from grid_process.helpers import make_accessor, reproject
from grid_process import data_model as dm
from grid_process.optimization import memoize
from grid_process.ds_utils import ds_to_da, cast_dtype
from grid_process.da_utils import da_to_ds

logger = logging.getLogger(__name__)


# Register xarray accessor so module functions can also be called as methods
# on xr.Dataset objects
@xr.register_dataset_accessor("grid_interp")
class GridInterp:
    """
    Xarray accessors class for spatial interpolation of gridded data
    """

    def __init__(self, ds: xr.Dataset):
        self._data = ds


dataset_accessor = partial(make_accessor, accessor_class=GridInterp)

POI_MODEL = CONFIG.data_model.poi
POI_DIM = POI_MODEL.dim
POI_COORD_COL = POI_MODEL.coord_from
SRC_ATTRS_KEEP = ("source", "title", "institution", "history", "references", "comment")
SWISS_CRS = ("EPSG:2056", "EPSG:21781")

InterpMethods = (
    Literal[
        "idw",
        "temp",
        "temp_fixgrad",
        "nearest_alt",
        "snowlim",
        "nearest_3D",
        "linear_3D",
        "idw_3D",
        "upscaled",
        "upscaled_3D",
    ]
    | Literal["nearest", "linear", "cubic"]
)


def interp_temp(  # pylint: disable=too-many-locals
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    z_fact: int = 25,
    k: int = 36,
    weight_power: float | None = None,
    grad_only: bool = False,
    chunk_size: int = 20,
) -> np.ndarray:
    """
    Interpolation from grid to POIs for temperature data, based on local
    soundings. The grid does not need to be regular. The sounding curves are
    fitted on the k nearest neighbors considering a distance that more heavily
    penalizes elevation differences for ``z_fact`` > 1.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 3)
    z_fact : positive int, optional
        factor to multiply the z coordinates to find nearest neighbors based on a
        non-euclidean distance that penalizes more altitude differences. Default
        value is 50. A value of 1 will use regular euclidean distances.
    k : positive int, optional
        number of considered nearest neighbors for the local sounding. Default
        is 36
    weight_power : positive float, optional
        a weight vector is passed to the sounding fit according to
        ``w = d^(-weight_power)``, with d being the (non-euclidean) distances between
        the considered POI and the k nearest grid points. The greater is the
        ``weight_power``, the stronger is the penalization with distance. A
        ``weight_power`` of 0 is equivalent to None (i.e. same weights for all
        points). By default, gives same weight for all points
    grad_only : bool, optional
        if True, only uses the local gradient to correct the nearest gridpoint
        temperature based on the altitude difference with the POI. Default is
        False
    chunk_size : positive int, optional
        number of POIs processed per sounding batch. Smaller values reduce peak
        memory (dominant cost: ``chunk_size × k × n_trail × 12`` bytes) and
        keep the working set in L3 cache, which is typically faster when
        ``n_trail`` is large. Larger values reduce loop overhead and are better
        for small trailing dimensions or when n_pois is small. Default is 20.

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array

    Raises
    ------
    ValueError : Raised if the orography is flat, as no sounding can be estimated.

    See also
    --------
    :func:`interp2poi`

    Note
    ----
    For POIs with altitude higher than 300m above the highest gridpoint or lower
    than 500m below the lowest gridpoint in the local neighborhood used for the fit,
    the fixgrad method is used with default gradient of 0.006 [K / m].
    """

    # make sure k is smaller or equal to the number of grid points
    k = min(k, len(coords_flat))

    # rescale z coords
    coords_flat_zfact = coords_flat.copy()
    pois_coords_zfact = pois_coords.copy()
    coords_flat_zfact[:, -1] *= z_fact
    pois_coords_zfact[:, -1] *= z_fact

    # find indexes and distances of k nearest neighbors in the non-euclidean dist
    dist_nn, ind_nn = nn_lookup_cached(coords_flat_zfact, pois_coords_zfact, k)

    # preallocate output array
    data_dtype = data_flat.dtype
    poi_data = np.empty((pois_coords.shape[0],) + data_flat.shape[1:], dtype=data_dtype)

    # gather neighbor altitudes for all POIs at once (small: n_pois × k floats)
    all_alt = coords_flat[ind_nn, -1]  # (n_pois, k)

    if weight_power is not None:
        # weights for the sounding fit according to the distance
        dist_weight: np.ndarray | None = _compute_dist_weight(dist_nn, weight_power, data_dtype)
    else:
        dist_weight = None

    poi_alts = pois_coords[:, -1]

    chunk_size = max(1, chunk_size)

    if grad_only:
        # grad_only always uses a deg=1 fit regardless of altitude position, so
        # the deg3/deg1 split is unnecessary. Only min/max quantiles are needed
        # to classify fixgrad vs sounding POIs.
        alt_minmax = np.quantile(all_alt, [0, 1], axis=1)  # shape (2, n_pois)
        fixgrad_mask = (poi_alts < alt_minmax[0] - 600) | (poi_alts > alt_minmax[1] + 200)
        sounding_groups = [(np.nonzero(~fixgrad_mask)[0], 1, True)]
    else:
        # classify each POI into three groups: fixgrad, deg1 sounding, deg3 sounding
        alt_q = np.quantile(all_alt, [0, 0.15, 0.85, 1], axis=1)  # shape (4, n_pois)
        fixgrad_mask = (poi_alts < alt_q[0] - 600) | (poi_alts > alt_q[-1] + 200)
        sounding_mask = ~fixgrad_mask
        deg3_mask = sounding_mask & (alt_q[1] < poi_alts) & (poi_alts < alt_q[2])
        deg1_mask = sounding_mask & ~deg3_mask
        sounding_groups = [
            (np.nonzero(deg1_mask)[0], 1, False),
            (np.nonzero(deg3_mask)[0], 3, False),
        ]

    # fixgrad group: only needs the nearest neighbor per POI (same for both branches)
    fg_idx = np.nonzero(fixgrad_mask)[0]
    if fg_idx.size > 0:
        poi_data[fg_idx] = _fixgrad(
            data_flat[ind_nn[fg_idx, 0]],
            all_alt[fg_idx, 0],
            poi_alts[fg_idx],
        )

    # sounding groups: process in memory-bounded chunks
    for idx, deg, go in sounding_groups:
        for start in range(0, idx.size, chunk_size):
            chunk_idx = idx[start : start + chunk_size]
            chunk_w = dist_weight[chunk_idx] if dist_weight is not None else None
            poi_data[chunk_idx] = _sounding_batch(
                data_flat[ind_nn[chunk_idx]],
                all_alt[chunk_idx],
                poi_alts[chunk_idx],
                deg=deg,
                w=chunk_w,
                grad_only=go,
            )

    return poi_data


def _normalize_altitudes(
    altitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize altitudes per POI to improve numerical conditioning of the
    polynomial fit.

    Parameters
    ----------
    altitudes : numpy.ndarray, shape (n_sub, k)
        neighbor altitudes for each POI

    Returns
    -------
    alt_norm : numpy.ndarray, shape (n_sub, k)
        normalized altitudes with zero mean and unit standard deviation per POI
    alt_mean : numpy.ndarray, shape (n_sub, 1)
        per-POI altitude mean used for normalization
    alt_std : numpy.ndarray, shape (n_sub, 1)
        per-POI altitude standard deviation used for normalization
    """

    alt_mean = altitudes.mean(axis=1, keepdims=True)
    alt_std = altitudes.std(axis=1, keepdims=True).clip(1e-10)
    return (altitudes - alt_mean) / alt_std, alt_mean, alt_std


def _build_vandermonde(alt_norm: np.ndarray, deg: int) -> np.ndarray:
    """
    Build a batched Vandermonde matrix from normalized altitudes.

    Parameters
    ----------
    alt_norm : numpy.ndarray, shape (n_sub, k)
        normalized neighbor altitudes
    deg : positive int
        polynomial degree

    Returns
    -------
    V : numpy.ndarray, shape (n_sub, k, deg+1)
        Vandermonde matrix with columns [1, x, x^2, ..., x^deg]
    """

    exponents = np.arange(deg + 1, dtype=np.float64)
    return alt_norm[:, :, None] ** exponents


def _apply_weights(vandermonde: np.ndarray, values: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply per-POI weights to the Vandermonde matrix and right-hand side.

    Parameters
    ----------
    vandermonde : numpy.ndarray, shape (n_sub, k, deg+1)
        Vandermonde matrix
    values : numpy.ndarray, shape (n_sub, k, n_trail)
        flattened data values
    w : numpy.ndarray, shape (n_sub, k) or None
        weight vector per POI; if None, uniform weights are used

    Returns
    -------
    vw : numpy.ndarray, shape (n_sub, k, deg+1)
        weight-scaled Vandermonde matrix
    yw : numpy.ndarray, shape (n_sub, k, n_trail)
        weight-scaled right-hand side
    """

    if w is not None:
        ww = w[:, :, None].astype(np.float64, copy=False)
        return vandermonde * ww, values * ww
    return vandermonde, values


def _qr_polyfit_batch(vw: np.ndarray, yw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a batch of weighted least-squares problems via QR decomposition and
    detect rank-deficient systems.

    Parameters
    ----------
    vw : numpy.ndarray, shape (n_sub, k, deg+1)
        weighted Vandermonde matrix
    yw : numpy.ndarray, shape (n_sub, k, n_trail)
        weighted right-hand side

    Returns
    -------
    coeffs : numpy.ndarray, shape (n_sub, deg+1, n_trail)
        polynomial coefficients; zero for rank-deficient systems
    rank_deficient : numpy.ndarray, shape (n_sub,), dtype bool
        True for systems where the R diagonal indicates near-singularity
    """

    q_mat, r_mat = np.linalg.qr(vw)
    qt_b = q_mat.swapaxes(-2, -1) @ yw  # (n_sub, deg+1, n_trail)
    diag_abs = np.abs(np.diagonal(r_mat, axis1=-2, axis2=-1))  # (n_sub, deg+1)
    rank_ok = diag_abs.min(axis=-1) >= 1e-10 * diag_abs.max(axis=-1)
    coeffs = np.zeros_like(qt_b)
    ok_idx = np.nonzero(rank_ok)[0]
    if ok_idx.size > 0:
        coeffs[ok_idx] = np.linalg.solve(r_mat[ok_idx], qt_b[ok_idx])
    return coeffs, ~rank_ok


def _eval_poly_batch(coeffs: np.ndarray, poi_norm: np.ndarray, deg: int) -> np.ndarray:
    """
    Evaluate a batch of polynomials at normalized POI altitudes.

    Parameters
    ----------
    coeffs : numpy.ndarray, shape (n_sub, deg+1, n_trail)
        polynomial coefficients in the normalized altitude space
    poi_norm : numpy.ndarray, shape (n_sub,)
        normalized target altitudes for each POI
    deg : positive int
        polynomial degree

    Returns
    -------
    result : numpy.ndarray, shape (n_sub, n_trail)
        evaluated polynomial values
    """

    exponents = np.arange(deg + 1, dtype=np.float64)
    vander_poi = poi_norm[:, None] ** exponents  # (n_sub, deg+1)
    return (vander_poi[:, None, :] @ coeffs)[:, 0, :]  # (n_sub, n_trail)


def _sounding_batch(  # pylint: disable=too-many-locals
    data_array: np.ndarray,
    altitudes: np.ndarray,
    poi_alts: np.ndarray,
    deg: int = 3,
    w: np.ndarray | None = None,
    grad_only: bool = False,
) -> np.ndarray:
    """
    Returns predicted temperature values for a batch of POIs according to the
    local sounding, using a batched QR decomposition.

    Each POI has its own neighborhood and independent polynomial fit. All fits
    are computed simultaneously via ``numpy.linalg.qr`` and
    ``numpy.linalg.solve`` to avoid per-POI Python-level dispatch.

    Parameters
    ----------
    data_array : numpy.ndarray, shape (n_sub, k, ...)
        contains the temperature values of the k nearest neighbor grid points
        for each of the n_sub POIs in this batch
    altitudes : numpy.ndarray, shape (n_sub, k)
        contains the altitudes of the k nearest neighbor grid points for each POI
    poi_alts : numpy.ndarray, shape (n_sub,)
        altitude of each POI in this batch
    deg : positive int
        degree of the polynomial fit. Only used if ``grad_only=False``
    w : numpy.ndarray, shape (n_sub, k) or None
        per-POI weight vectors for the polynomial fit. If None, uniform weights
        are used for all points
    grad_only : bool
        if True, only uses the local gradient to correct the temperature of the
        nearest gridpoint based on the altitude difference with the POI.
        In this case, the gradient is bound to [-0.01, 0.03] K m-1. Default is False

    Returns
    -------
    fitted_val : numpy.ndarray, shape (n_sub, ...)
        array of fitted temperatures for each POI in the batch

    See also
    --------
    :func:`interp_temp`
    """

    n_sub = data_array.shape[0]
    trail_shape = data_array.shape[2:]
    n_trail = int(np.prod(trail_shape)) if trail_shape else 1

    # flatten trailing dims and cast to float64 for numerical stability
    values = data_array.reshape(n_sub, data_array.shape[1], n_trail, order="C").astype(np.float64, copy=False)
    alt_norm, alt_mean, alt_std = _normalize_altitudes(altitudes)

    if grad_only:
        vandermonde = _build_vandermonde(alt_norm, 1)
        vw, yw = _apply_weights(vandermonde, values, w)
        coeffs, _ = _qr_polyfit_batch(vw, yw)
        # convert slope from normalized space back to original altitude units
        slope = (coeffs[:, 1] / alt_std[:, 0, None]).clip(-0.01, 0.03)
        fitted_val = values[:, 0] + slope * (poi_alts[:, None] - altitudes[:, 0, None])
    else:
        vandermonde = _build_vandermonde(alt_norm, deg)
        vw, yw = _apply_weights(vandermonde, values, w)
        coeffs, rank_deficient = _qr_polyfit_batch(vw, yw)

        poi_norm = (poi_alts - alt_mean[:, 0]) / alt_std[:, 0]
        fitted_val = _eval_poly_batch(coeffs, poi_norm, deg).reshape(n_sub, *trail_shape)

        # overwrite rank-deficient results with a deg=1 fallback. If deg is already 1, raise an error instead.
        bad_idx = np.nonzero(rank_deficient)[0]
        if bad_idx.size > 0:
            if deg == 1:
                raise ValueError(
                    f"deg=1 sounding is rank-deficient for {bad_idx.size} POI(s): "
                    "all k neighbor altitudes are identical. Check the surface altitude coordinate."
                )
            fitted_val[bad_idx] = _sounding_batch(
                data_array[bad_idx],
                altitudes[bad_idx],
                poi_alts[bad_idx],
                deg=1,
                w=w[bad_idx] if w is not None else None,
                grad_only=False,
            )

    return fitted_val.reshape((n_sub,) + trail_shape)


def interp_temp_fixgrad(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    r_search: float = 2,
    z_weight: int = 200,
    grad: float = 0.006,
) -> np.ndarray:
    """
    Interpolation from grid to POIs for temperature data by taking the
    surrounding gridpoint with the most similar altitude and further correct
    for altitude difference with a fixed gradient.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 3)
    r_search : positive float, optional
        radius in model gridpoints that define the number of neighbors included
        in the search. Default is 4
    z_weight : positive int, optional
        weight for the z axis in the distance computation of the "most adequat"
        gridpoint. ``d = horizontal_dist + z_weight * vertical_dist``.
        Default value is 200
    grad : positive float
        gradient in [K / m] to correct for altitude difference between POI and
        nearest grid point. Default is 0.006 [K / m]

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array.

    See also
    --------
    :func:`interp2poi`
    """

    ind = _grid_assignation(coords_flat, pois_coords, r_search, z_weight)

    return _fixgrad(
        data_flat[ind,],
        coords_flat[ind, -1],
        pois_coords[:, -1],
        grad,
    )


def _fixgrad(
    best_gp_data: np.ndarray,
    best_gp_alt: np.ndarray,
    pois_alt: np.ndarray,
    grad: float = 0.006,
) -> np.ndarray:
    """
    Correct temperature based on altitudinal differences and a fix gradient.

    Parameters
    ----------
    best_gp_data : numpy.ndarray, shape (n_pois, ...)
        contains the temperature values of the most appropriate gridpoint for
        each POI
    best_gp_alt : numpy.ndarray, shape (n_pois,)
        contains the altitude of the most appropriate gridpoint for each POI.
    pois_alt : numpy.ndarray, shape (n_pois,)
        contains the altitude of each POI
    grad : positive float
        gradient in [K / m] to correct for altitude difference between POI and
        nearest grid point. Default is 0.006 [K / m]

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array of altitude corrected temperatures

    See also
    --------
    :func:`interp_temp_fixgrad`
    """

    ndim_trail = best_gp_data.ndim - 1
    delta_t = (best_gp_alt - pois_alt) * grad
    delta_t = delta_t.astype(best_gp_data.dtype, copy=False)

    return best_gp_data + delta_t.reshape(delta_t.shape + (1,) * ndim_trail)


def interp_nearest_alt(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    r_search: float = 2,
    z_weight: int = 500,
) -> np.ndarray:
    """
    Interpolation from grid to POIs for elevation dependent data. The method finds
    the nearest grid point according to a distance that use a much larger weight
    along the z axis. For wind data for instance, grid points at similar altitudes
    are preferred. The grid does not need to be regular.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 3)
    r_search : positive float, optional
        radius in model gridpoints that define the number of neighbors included
        in the search. Default is 4
    z_weight : positive int, optional
        weight for the z axis in the distance computation of the "most adequat"
        gridpoint. ``d = horizontal_dist + z_weight * vertical_dist``.
        Default value is 500

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array

    See also
    --------
    :func:`interp2poi`
    """

    ind = _grid_assignation(coords_flat, pois_coords, r_search, z_weight)

    return data_flat[ind,]


def _grid_assignation(
    coords_flat: np.ndarray,
    pois_coords: np.ndarray,
    r_search: float,
    z_weight: int,
    deltah_min: int = 50,
) -> np.ndarray:
    """
    Returns the index of the "most appropriate" gridpoint based on a distance
    metric that gives a much greater weight to vertical distances. This is the
    standard used for verification in COSMO.
    """

    k = round(np.pi * r_search**2)
    # find indexes and distances of nearest neighbor grids for all POIs
    dist_nn_horz, ind_nn = nn_lookup_cached(coords_flat[:, :2], pois_coords[:, :2], k)

    dist_nn_vert = np.abs(coords_flat[ind_nn, -1] - pois_coords[:, [-1]])
    dist_weighted = dist_nn_horz + z_weight * (dist_nn_vert - deltah_min).clip(0)

    return ind_nn[np.arange(ind_nn.shape[0]), np.argmin(dist_weighted, axis=1)]


def interp_idw(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    k: int = 4,
    weight_power: float = 1.0,
) -> np.ndarray:
    """
    Performs interpolations from grid to POIs using inverse distance
    weightning.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids,m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 2 or 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 2 or 3)
    k : positive int, optional
        number of considered nearest neighbors. Default is 4
    weight_power : positive float, optional
        the interpolated value is the weighted average of the k nearest grid
        points accordi g to ``w = d^(-weight_power)``, with d being the
        distance between the considered POI and the gird points. The greater is
        the ``weight_power``, the stronger is the penalization with distance.
        A ``weight_power`` of 0 produces a normal average of the k nearest grid
        points

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array

    See also
    --------
    :func:`interp2poi`
    """

    # find indexes and distances of the k nearest neighbor grids for all POIS
    dist_nn, ind_nn = nn_lookup_cached(coords_flat, pois_coords, k)

    # compute the weights
    dist_weight = _compute_dist_weight(dist_nn, weight_power, data_flat.dtype)

    # vectorized weigthed average
    ndim_trail = data_flat.ndim - 1
    poi_data = np.sum(
        data_flat[ind_nn,] * dist_weight.reshape(dist_weight.shape + (1,) * ndim_trail),
        axis=1,
    )
    return poi_data


def interp_upscaled(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    k: int = 4,
    agg_func: Callable = np.max,
) -> np.ndarray:
    """
    Perform interpolations from grid to POIs by computing a group statistic
    over the k nearest neighbors (e.g. upscaling).

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 2 or 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 2 or 3).
    k : positive int, optional
        number of considered nearest neighbors. Default is 4
    agg_func : callable, optional
        specify the spatial aggregation function to use (i.e. numpy.mean,
        numpy.max, numpy.min, numpy.std). Default is numpy.max.

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array

    See also
    --------
    :func:`interp2poi`
    """

    # find indexes and distances of the k nearest neighbor grids for all POIS
    _, ind_nn = nn_lookup_cached(coords_flat, pois_coords, k)

    # group aggregation
    return agg_func(
        data_flat[ind_nn,],
        axis=1,
    )


def interp_nearest(coords_flat: np.ndarray, data_flat: np.ndarray, pois_coords: np.ndarray) -> np.ndarray:
    """
    Performs nearest neighbor interpolation from grid to POIs.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 2 or 3).
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 2 or 3).

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array

    See also
    --------
    :func:`interp2poi`
    """

    # find indexes and distances of the k nearest neighbor grids for all POIS
    _, ind_nn = nn_lookup_cached(coords_flat, pois_coords, 1)

    return data_flat[ind_nn]


def interp_snowlim(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    k: int = 36,
    d_max: float = 40000,
) -> np.ndarray:
    """
    Special interpolation routine for snow limit. Snow limits below model
    topography are undefined, resulting in datasets with randomly distributed
    NaNs. This routine checks for the nearest non-NaN grid points. If this
    point is further away than d_max, the snowlimit is set to a value between
    0 and 200 [m] depending on the POI altitude. Otherwise, the minimum between
    the nearest non-NaN grid point and the minimum altitude of NaN grid points
    among the k nearest neighbors is taken.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        Array containing the flattened model data with various trailing
        dimension.
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 3)
    k : positive int
        Number of considered nearest neighbors to check for altitude of NaN
        grid points.
    d_max : positive float
        if all grid points within a radius of d_max meter from the POI are NaNs
        the snow limit is set to a value between 0 and 200 [m] depending on
        the POI altitude

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array

    See also
    --------
    :func:`interp2poi`
    """

    data_shape = data_flat.shape
    n_pois = len(pois_coords)

    poi_dummy = _fillna_alt(  # NaNs filled with model topo
        coords_flat, data_flat, pois_coords, min(data_shape[0], k)
    )

    data_flat = data_flat.reshape((data_shape[0], -1), order="C")

    # array with default values when there is only NaNs within distance d_max
    poi_data = (pois_coords[:, 2] - 200).clip(0, 200).astype(data_flat.dtype, copy=False)
    poi_data -= poi_data % 50  # floor to multiple of 50
    poi_data = np.broadcast_to(poi_data[:, None], (n_pois, data_flat.shape[-1])).copy()

    for idx, array in enumerate(data_flat.T):
        ind = ~np.isnan(array)
        if ind.any():
            dist_nn, ind_nn = nn_lookup(coords_flat[ind, :2], pois_coords[:, :2], 1)
            bool_dist = dist_nn < d_max
            poi_data[bool_dist, idx] = array[ind][ind_nn[bool_dist]]

    poi_data = poi_data.reshape((n_pois,) + data_shape[1:], order="C")

    return np.minimum(poi_data, poi_dummy)


def _fillna_alt(coords_flat: np.ndarray, data_flat: np.ndarray, pois_coords: np.ndarray, k: int) -> np.ndarray:
    """
    Interp to POIs with nearest neighbor and fill NaNs by simply taking the
    altitude of the lowest grid point among the k nearest neighbors that have
    an undefined snow limit.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 3)
    k : positive int
        number of considered nearest neighbors to check for altitude of NaN

    Returns
    -------
    data_poi : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values/model topography with same
        trailing dimensions as the input ``data_flat`` array.

    See also
    -------
    :func:`interp_snowlim`
    """

    ndim_trail = data_flat.ndim - 1
    _, ind_nn = nn_lookup(coords_flat[:, :2], pois_coords[:, :2], k)
    data_poi = data_flat[ind_nn]
    ind_na = np.isnan(data_poi)
    alt_poi = coords_flat[ind_nn, -1].reshape(ind_nn.shape + (1,) * ndim_trail)
    alt_poi = np.broadcast_to(alt_poi, data_poi.shape)
    alt_poi = np.amin(alt_poi, axis=1, initial=10000, where=ind_na)
    data_poi = data_poi[:, 0]
    data_poi[ind_na[:, 0]] = alt_poi[ind_na[:, 0]]

    return data_poi


@dataset_accessor
def interp2poi(
    data: xr.Dataset,
    poi_info: gpd.GeoDataFrame,
    method: InterpMethods,
    try_reg: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Interpolates gridded spatial data to a set of Points of Interest (POIs).

    This function provides a unified interface for interpolating values from regular or unstructured spatial grids
    to arbitrary POI locations, supporting a variety of interpolation methods and data models. It replaces the
    original spatial grid dimensions in `data` with a POI dimension, containing interpolated values for each POI.

    Parameters
    ----------
    data : xarray.Dataset
        The source grid dataset. Supported data models include:

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

        For methods requiring altitude, ensure a surface altitude coordinate is present; extend supported names via
        GRID_PROCESS__data_model__grid__surf_alt.
    poi_info : geopandas.GeoDataFrame
        POI locations and attributes. Must have geometry and a defined CRS.
        For methods requiring altitude, a POI altitude column or z-component in geometry (in units of meters) is needed.
        Extend supported POI altitude column names via:

          - export GRID_PROCESS__data_model__poi__alt_from (e.g., '["height_masl", "altitude", "dummy_height"]')

        The column `name` is used for the output POI coordinate by default; override via:

          - export GRID_PROCESS__data_model__poi__coord_from (e.g., dummy_name)
    method : str
        Interpolation method. Supported options include:

            {'nearest', 'linear', 'cubic', 'idw', 'temp', 'temp_fixgrad', 'nearest_alt', 'snowlim',
            'nearest_3D', 'linear_3D', 'idw_3D', 'upscaled', 'upscaled_3D'}.

        3D methods include altitude in neighbor selection. For unstructured grids, prefer 'idw' with k=4 and
        weight_power=2 over 'linear' for performence reasons.
    try_reg : bool, default True
        If True, attempts regular grid interpolation for structured grids with 1D coordinates when ``method`` is
        'nearest' or 'linear'. Regular grid interpolation is typically more efficient for large grids, but may be less
        so for small grids with many trailing dimensions.
    **kwargs : Any
        Additional arguments passed to the specific interpolation sub-function, named `interp_<method>`.

    Returns
    -------
    xr.Dataset
        Dataset where original spatial dimensions are replaced with a POI dimension containing interpolated values.
        The output POI dimension name can be overridden via GRID_PROCESS__data_model__poi__dim (e.g. station).

    See also
    --------
    :func:`interp_temp`, :func:`interp_temp_fixgrad`, :func:`interp_nearest_alt`,
    :func:`interp_idw`, :func:`interp_upscaled`, :func:`interp_nearest`,
    :func:`interp_snowlim`, :func:`scipy.interpolate.griddata`, :meth:`xarray.Dataset.interp`

    Notes
    -----
    - Grid points containing only NaNs across all dimensions are excluded from interpolation, enabling easy masking
      (e.g., excluding water pixels with `ds.where(ds.fr_land > 0.5).interp2poi(..., try_reg=False)`).
    - For regular grid interpolation, all NaN grid points are kept.
    - Environment variables allow flexible support for custom grid and POI data models.
    """

    def _fullnan_ds(data: xr.DataArray) -> xr.Dataset:
        """
        Returns a dataset full of NaNs with correct dimensions
        """

        da = (
            data.isel({dim: 0 for dim in spatial_dims}, drop=True)
            .expand_dims({POI_DIM: poi_info[POI_COORD_COL].to_numpy(dtype=str)})
            .copy()
        )
        da.attrs = _add_attrs(data.attrs)
        da[:] = np.nan
        return da_to_ds(da, vars_attrs)

    def _array2ds(data: xr.DataArray) -> xr.Dataset:
        """
        Converts an array with interpolated values into a dataset with correct
        dimensions, coordinates and attributes
        """

        da = xr.DataArray(
            poi_data,
            dims=(POI_DIM,) + data.dims[len(spatial_dims) :],
            coords={
                POI_DIM: poi_info[POI_COORD_COL].to_numpy(dtype=str),
                **data.isel({dim: 0 for dim in spatial_dims}, drop=True).coords,
            },
            attrs=_add_attrs(data.attrs),
        )
        return da_to_ds(da, vars_attrs)

    spatial_dims = dm.get_spatial_dims(data)
    data = dm.set_grid_mapping(data)
    cond_reg = method in ("nearest", "linear") and len(spatial_dims) > 1 and set(spatial_dims).issubset(data.coords)
    if try_reg and cond_reg:
        return _interp_regular(data, spatial_dims, poi_info, method)  # type: ignore[arg-type]

    data = slice_boundingbox(data, poi_info, method, spatial_dims, **kwargs)
    data, vars_attrs = ds_to_da(data)  # type: ignore[assignment]
    data = data.transpose(*spatial_dims, ...)
    coords_flat, data_flat, pois_coords = _flatten_concat(
        data,  # type: ignore[arg-type]
        spatial_dims,
        poi_info,
        method,
    )

    if len(coords_flat) == 0:
        return _fullnan_ds(data)  # type: ignore[arg-type]

    if method in INTERP_FUNC:
        poi_data = INTERP_FUNC[method](coords_flat, data_flat, pois_coords, **kwargs)  # type: ignore[operator]

    elif method in ("linear", "linear_3D", "cubic"):
        poi_data = griddata(coords_flat, data_flat, pois_coords, method.rsplit("_")[0])
        poi_data = poi_data.astype(data_flat.dtype, copy=False)
    else:
        raise ValueError(f"Unknown interpolation method {method!r}")

    return _array2ds(data)  # type: ignore[arg-type]


def _interp_regular(  # pylint: disable=too-many-locals
    data: xr.Dataset,
    spatial_dims: tuple[str, str],
    poi_info: gpd.GeoDataFrame,
    method: Literal["nearest", "linear"] = "linear",
) -> xr.Dataset:
    """
    Wrapper for interpolation from a regular grid to POIs.
    Handle reprojection of the points to the native crs of the input grid.
    Also assigns coordinate labels to the output poi dimension according to the
    POI_COORD_COL column of ``poi_info``. Used in :func:`interp2poi`
    """

    def _interp_cast_dtype(
        method: Literal["nearest", "linear"],
    ) -> xr.Dataset:
        """
        Wrapper for xarray.Dataset.interp to return data in the same dtype as
        input.
        """

        poi_data = data.interp({x_dim: pois_coords[0], y_dim: pois_coords[1]}, method, assume_sorted=True).drop_vars(
            spatial_dims, errors="ignore"
        )

        cast_dtype(poi_data, data)

        return poi_data

    data = _drop_coords_reg(data, spatial_dims)
    y_dim, x_dim = spatial_dims
    data = _y_increasing(data, y_dim)

    # Reproject POIs coordinates in the grid CRS
    pois_coords = _convert_poi(poi_info, dm.get_grid_crs(data), data[y_dim].dtype)
    poi_dim_coord = poi_info[POI_COORD_COL].to_numpy(dtype=str)
    pois_coords = (
        xr.DataArray(pois_coords[0], coords=[(POI_DIM, poi_dim_coord)]),
        xr.DataArray(pois_coords[1], coords=[(POI_DIM, poi_dim_coord)]),
    )

    if method == "nearest":
        dx, dy = np.diff(data[x_dim].values), np.diff(data[y_dim].values)
        cst_spacing = isclose(dx.min(), dx.max(), rel_tol=1e-4) and isclose(dy.min(), dy.max(), rel_tol=1e-4)
        dtype_bool = not all(data[var].dtype in (np.float32, np.float64) for var in data)
        chunk_bool = len(data.chunks) > 0
        if cst_spacing and (chunk_bool or dtype_bool):
            grid_spacing = float(abs(dx[0])), float(abs(dy[0]))
            poi_data = _interp_nearest_regular(data, spatial_dims, grid_spacing, pois_coords)
        else:
            poi_data = _interp_cast_dtype("nearest")
    else:
        poi_data = _interp_cast_dtype(method)

    poi_data.attrs = _add_attrs(data.attrs)

    return poi_data.transpose(POI_DIM, ...)


def _y_increasing(ds: xr.Dataset, y_dim: str) -> xr.Dataset:
    """
    Reverse along the y dimension if coordinates are decreasing instead of increasing.
    This allows to use ``assume_sorted=True`` in :meth:`xarray.Dataset.interp` which is significantly faster.
    """

    if dm.is_decreasing1d(ds[y_dim]):
        ds = ds.isel({y_dim: slice(None, None, -1)})

    return ds


def _drop_coords_reg(data: xr.Dataset, spatial_dims: tuple[str, str]) -> xr.Dataset:
    """
    Drop non-dimensional coordinate variables that have spatial dimensions.
    """

    notdim_coords = set(data.coords) - set(data.dims)
    spd = set(spatial_dims)
    to_drop = [coord for coord in notdim_coords if spd.issubset(data[coord].dims)]

    return data.drop_vars(to_drop)


def _interp_nearest_regular(
    data: xr.Dataset,
    spatial_dims: tuple[str, str],
    grid_spacing: tuple[float, float],
    pois_coords: tuple[xr.DataArray, xr.DataArray],
) -> xr.Dataset:
    """
    Perform fast nearest neighbor interpolation from a regular grid with
    constant spacing along each dimension to POIs. Preserve dtype
    so the memory footprint is limited compared to xarray.Dataset.interp which
    cast to numpy.float64.

    Note
    ----
    If the spacing along a dimension is not constant, results may be wrong.
    """

    y_dim, x_dim = spatial_dims
    data = data.sel(
        {
            x_dim: slice(
                pois_coords[0].min() - grid_spacing[0],
                pois_coords[0].max() + grid_spacing[0],
            ),
            y_dim: slice(
                pois_coords[1].min() - grid_spacing[1],
                pois_coords[1].max() + grid_spacing[1],
            ),
        }
    )
    data = data.chunk()  # cast to dask
    grid_coords = (data[x_dim].values, data[y_dim].values)
    hkey = (
        tuple(pois_coords[0].poi.values.tolist()),
        pois_coords[0].values.tobytes(),
        pois_coords[1].values.tobytes(),
        grid_coords[0].tobytes(),
        grid_coords[1].tobytes(),
    )
    # pylint: disable-next=unbalanced-tuple-unpacking  # returns one index array per spatial dimension
    indx, indy = nn_lookup_regular(grid_coords, grid_spacing, pois_coords, hkey=hkey)  # type: ignore[call-arg]
    return data.isel({x_dim: indx, y_dim: indy}).drop_vars(spatial_dims, errors="ignore")


@memoize(CONFIG.cache_size.nn_lookup)
def nn_lookup(grid_coords: np.ndarray, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the k nearest grid points and corresponding distances for a set of
    POIs.

    Parameters
    ----------
    grid_coords : numpy.ndarray, shape (n, m)
        array of flattened grid coordinates. N is the number of model grid
        points, m the number of coordinates dimensions (i.e. 2 or 3)
    points : numpy.ndarray, shape (p, m)
        array of POIs coordinates. P is the number of POIs, m the number of
        coordinates dimensions (i.e. 2 or 3)
    k : positive int
        number of considered nearest neighbors

    Returns
    -------
    dd : numpy.ndarray, shape (p, k)
        array of nearest neighbor distances.
    ii : numpy.ndarray, shape (p, k)
        array of nearest neighbor grid indexes

    See also
    --------
    :func:`nn_lookup_cached`
    """

    tree = KDTree(grid_coords)
    dd, ii = tree.query(points, k, workers=-1)

    return dd.astype(points.dtype, copy=False), ii


def nn_lookup_cached(grid_coords: np.ndarray, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to :func:`nn_lookup` function to use it with a cache
    """

    hkey = (grid_coords.tobytes(), points.tobytes(), k)
    return nn_lookup(grid_coords, points, k, hkey=hkey)  # type: ignore[call-arg]


@memoize(CONFIG.cache_size.nn_lookup_regular)
def nn_lookup_regular(
    grid_coords: tuple[np.ndarray, np.ndarray],
    grid_spacing: tuple[float, float],
    points: tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray],
) -> list[np.ndarray | xr.DataArray]:
    """
    Find the nearest grid point for a set of POIs when the source
    grid is regular and the spacing constant along each dimension.

    Parameters
    ----------
    grid_coords : tuple of m 1-D arrays
        contains flattened grid coordinates arrays for all m dimensions
    grid_spacing : tuple of m floats
        contains the grid spacing for all m dimensions
    points : tuple of m 1-D arrays (or xarray.DataArray with dim poi)
        contains POIs coordinates for all m dimensions. It assumes the same
        order of dimensions as in ``grid_coords``

    Returns
    -------
    index : list of m 1-D arrays (or xarray DataArray with dimension `poi`)
        Used to index the dataset in each spatial dimension. The output indexes
        have length equals to the number of points
    """

    if len(points) != len(grid_coords):
        raise ValueError("The number of grid and points dimensions does not match!")
    grid_index = []
    for n, dim_coords in enumerate(grid_coords):
        dim_spacing = grid_spacing[n]
        dim_origin = np.min(dim_coords) - dim_spacing / 2
        dim_index = (points[n] - dim_origin) // dim_spacing
        dim_index = np.clip(dim_index, 0, dim_coords.size - 1).astype(int)
        grid_index.append(dim_index)
    return grid_index


def _compute_dist_weight(dist: np.ndarray, power: float, dtype: np.dtype) -> np.ndarray:
    """
    Compute weights based on the inverse distance. The weights scales to 1.
    """

    increm = np.mean(dist, axis=1, keepdims=True) / 50  # avoid division by 0
    w = 1 / np.power(dist + increm, power)
    w = w.astype(dtype, copy=False)
    return w / np.sum(w, axis=1, keepdims=True)


def _add_attrs(data_attrs: dict) -> dict:
    """
    Add attrs from source grid
    """

    return {key: data_attrs[key] for key in SRC_ATTRS_KEEP if key in data_attrs}


def _flatten_concat(
    data: xr.DataArray,
    spatial_dims: tuple[str, ...],
    poi_info: gpd.GeoDataFrame,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flattens the grid and concatenate columnwise `x` and `y` coordinates into a
    single 2D array. Also, reprojects the POIs coordinates if they are not in
    the same CRS than the model coordinates
    """

    data, grid_y, grid_x, grid_crs = _flatten_coords_crs(data, spatial_dims)

    if method in HEIGHT_METHODS:
        # include elevation
        surf_alt = dm.get_surf_alt(data)
        coords_flat = np.c_[
            data[grid_x].values,
            data[grid_y].values,
            data[surf_alt].values,
        ]

        if grid_crs not in SWISS_CRS:
            # project to CH1903+/LV95 to have units in [m]
            new_crs = CRS.from_string(SWISS_CRS[0])
            coords_flat[:, 0], coords_flat[:, 1] = _reproject_grid_cached(
                coords_flat[:, 0], coords_flat[:, 1], grid_crs, new_crs
            )
            grid_crs = new_crs
    else:
        coords_flat = np.c_[data[grid_x].values, data[grid_y].values]

    # Reproject POIs coordinates in the grid CRS
    pois_coords = _convert_poi(poi_info, grid_crs, coords_flat.dtype)
    if method in HEIGHT_METHODS:
        pois_coords_cat = np.c_[
            pois_coords[0],
            pois_coords[1],
            dm.get_poi_alt(poi_info),
        ]
    else:
        pois_coords_cat = np.c_[pois_coords[0], pois_coords[1]]

    if method == "snowlim":
        return coords_flat, data.values, pois_coords_cat

    # check for all NaN grid points
    ind = ~data.isnull().all(data.dims[1:]).values

    return coords_flat[ind], data.values[ind], pois_coords_cat


def _convert_poi(poi_info: gpd.GeoDataFrame, grid_crs: CRS, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """
    Cached reprojection of POIs coordinates in the grid CRS. Returns a tuple of xy
    coordinates with the specified dtype.
    """

    pois_coords = poi_info.get_coordinates().values
    pois_coords = pois_coords.astype(dtype, copy=False)

    poi_crs = poi_info.crs.to_2d()
    if grid_crs == poi_crs:
        return pois_coords[:, 0], pois_coords[:, 1]

    hkey = (pois_coords[:, 0].tobytes(), pois_coords[:, 1].tobytes(), grid_crs)
    return _reproject_poi(pois_coords, poi_crs, grid_crs, hkey=hkey)  # type: ignore[call-arg]


@memoize(CONFIG.cache_size.reproject_poi)
def _reproject_poi(coords: np.ndarray, src_crs: CRS, dst_crs: CRS) -> tuple[np.ndarray, np.ndarray]:
    """
    Reprojection of POIs coordinates that can be cached
    """

    x_coords, y_coords = reproject(coords[:, 0], coords[:, 1], src_crs, dst_crs)
    dtype = coords.dtype
    x_coords = x_coords.astype(dtype, copy=False)
    y_coords = y_coords.astype(dtype, copy=False)
    return x_coords, y_coords


@memoize(CONFIG.cache_size.reproject_grid)
def _reproject_grid(
    x_coords: np.ndarray, y_coords: np.ndarray, src_crs: CRS, dst_crs: CRS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reprojection of grid coordinates that can be cached
    """

    return reproject(x_coords, y_coords, src_crs, dst_crs)


def _reproject_grid_cached(
    x_coords: np.ndarray, y_coords: np.ndarray, src_crs: CRS, dst_crs: CRS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cached reprojection of grid coordinates
    """

    if dst_crs != src_crs:
        hkey = (x_coords.tobytes(), y_coords.tobytes(), x_coords.shape, dst_crs)
        x_coords, y_coords = _reproject_grid(  # type: ignore[call-arg]
            x_coords, y_coords, src_crs, dst_crs, hkey=hkey
        )

    return x_coords.astype(np.float64, copy=False), y_coords.astype(np.float64, copy=False)


@dataset_accessor
def interp2grid(
    data: xr.Dataset,
    dst_grid: xr.Dataset,
    method: InterpMethods,
    try_reg: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Interpolates data from a source grid to a destination grid, supporting regular and unstructured grids.

    This function provides a unified interface for grid-to-grid interpolation, supporting a variety of interpolation
    methods and data models. The output dataset contains the data of the source grid mapped onto the destination grid
    coordinates.

    Parameters
    ----------
    data : xarray.Dataset
        The source grid dataset. Supported data models include:

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

        For methods requiring altitude, ensure a surface altitude coordinate is present; extend supported names via
        GRID_PROCESS__data_model__grid__surf_alt.
    dst_grid : xarray.Dataset
        The destination grid to interpolate to. Same supported data models as the source grid, including conventions for
        dimensions and coordinates as well as spatial CRS.
        For methods requiring altitude, ensure a surface altitude coordinate is also present.
        The destination dataset may contain only spatial coordinates or also data variables which will be ignored.
    method : str
        Interpolation method. Supported options include:

            {'nearest', 'linear', 'cubic', 'idw', 'temp', 'temp_fixgrad', 'nearest_alt', 'snowlim',
            'nearest_3D', 'linear_3D', 'idw_3D', 'upscaled', 'upscaled_3D'}.

        3D methods include altitude in neighbor selection. For unstructured grids, prefer 'idw' with k=4 and
        weight_power=2 over 'linear' for performence reasons.
    try_reg : bool, default True
        If True, attempts regular grid interpolation for structured source grids with 1D coordinates when ``method`` is
        'nearest' or 'linear'. Regular grid interpolation is typically more efficient for large grids, but may be less
        so for small grids with many trailing dimensions.
    **kwargs : Any
        Additional arguments passed to the specific interpolation sub-function, named `interp_<method>`.

    Returns
    -------
    xr.Dataset
        Dataset with values from `data` interpolated onto the coordinates of `dst_grid`.

    See also
    --------
    :func:`interp_temp`, :func:`interp_temp_fixgrad`, :func:`interp_nearest_alt`,
    :func:`interp_idw`, :func:`interp_upscaled`, :func:`interp_nearest`,
    :func:`interp_snowlim`, :func:`scipy.interpolate.griddata`, :meth:`xarray.Dataset.interp`

    Notes
    -----
    - Not all interpolation methods scale efficiently with large input datasets or and/or destination grids.
    - Recommended methods: 'nearest', 'linear' (if the source grid is regular with 1D coordinates), 'idw' (with k=4),
      'temp_fixgrad', 'nearest_3D', 'nearest_alt'.
    - Environment variables allow flexible support for custom grid data models.
    """

    def _array2ds(data: xr.DataArray) -> xr.Dataset:
        """
        Converts an array with interpolated values into a dataset with correct
        dimensions, coordinates and attributes
        """

        da = xr.DataArray(
            interp_data.reshape(
                (
                    dst_grid.sizes[dst_spatial_dims[0]],
                    dst_grid.sizes[dst_spatial_dims[1]],
                )
                + data.shape[len(src_spatial_dims) :]
            ),
            dims=dst_spatial_dims + data.dims[len(src_spatial_dims) :],
            coords={
                **data.isel({dim: 0 for dim in src_spatial_dims}, drop=True).coords,
                **dst_grid.coords,
            },
            attrs=_combine_attrs(data.attrs, dst_grid.attrs),
        )
        return da_to_ds(da, vars_attrs)

    src_spatial_dims = dm.get_spatial_dims(data)
    dst_spatial_dims = dm.get_spatial_dims(dst_grid)
    data = dm.set_grid_mapping(data)
    dst_grid = dm.set_grid_mapping(dst_grid)
    dst_grid = dst_grid.drop_dims(set(dst_grid.dims) - set(dst_spatial_dims))

    cond_reg = (
        method in ("nearest", "linear") and len(src_spatial_dims) > 1 and set(src_spatial_dims).issubset(data.coords)
    )
    if try_reg and cond_reg:
        return _interp_grid_regular(
            data,
            src_spatial_dims,  # type: ignore[arg-type]
            dst_grid,
            dst_spatial_dims,  # type: ignore[arg-type]
            method,  # type: ignore[arg-type]
        )

    dst_grid = dst_grid.transpose(*dst_spatial_dims)

    data = slice_coarsen(data, dst_grid, method, src_spatial_dims, dst_spatial_dims, **kwargs)
    data, vars_attrs = ds_to_da(data)  # type: ignore[assignment]
    data = data.transpose(*src_spatial_dims, ...)
    coords_flat, data_flat, dst_coords_flat = _flatten_concat_grids(
        data,  # type: ignore[arg-type]
        src_spatial_dims,
        dst_grid,
        dst_spatial_dims,  # type: ignore[arg-type]
        method,
    )

    if method in INTERP_FUNC:
        interp_data = INTERP_FUNC[method](  # type: ignore[operator]
            coords_flat, data_flat, dst_coords_flat, **kwargs
        )

    elif method in ("linear", "linear_3D", "cubic"):
        interp_data = griddata(coords_flat, data_flat, dst_coords_flat, method.rsplit("_")[0])
        interp_data = interp_data.astype(data_flat.dtype, copy=False)
    else:
        raise ValueError(f"Unknown interpolation method {method!r}")

    return _array2ds(data)  # type: ignore[arg-type]


def _interp_grid_regular(
    data: xr.Dataset,
    src_spatial_dims: tuple[str, str],
    dst_grid: xr.Dataset,
    dst_spatial_dims: tuple[str, str],
    method: Literal["nearest", "linear"] = "linear",
) -> xr.Dataset:
    """
    Wrapper for interpolation from a regular grid to another one.
    Handle reprojection of the destination grid to the native crs of the source grid.
    The destination grid does not need to be regular. Used in :func:`interp2grid`.

    Note
    ----
    For very large grids in 'x' and 'y', the interpolation is much more efficient
    when the input and destination grids are on the same native CRS.
    """

    def _cond_dst_1d() -> bool:
        """
        Whether the destination coordinates should be kept as 1D (when the source and destinations grids are in the
        same CRS). Depending on the size of the grids, xarray "nearest" interpolation is much faster when destination
        coordinates are 2D.
        """

        data_trail_size = prod([size for dim, size in data.sizes.items() if dim not in src_spatial_dims])
        dst_sizes = dst_grid.sizes
        dim_ratio = (dst_sizes[dst_spatial_dims[1]] * dst_sizes[dst_spatial_dims[0]]) / data_trail_size
        bool_dst_1d = dim_ratio > 2000 or method == "linear"
        return bool_dst_1d and set(dst_spatial_dims).issubset(dst_grid.coords)

    def _reproject_dst_2d() -> tuple[xr.DataArray, xr.DataArray]:
        """
        Reproject destination 2D coordinates in the source grid CRS
        """

        x_coords_val = dst_grid[dst_coords_crs[1]].values
        y_coords_val = dst_grid[dst_coords_crs[0]].values

        if x_coords_val.ndim == 1:
            x_coords_val, y_coords_val = np.meshgrid(x_coords_val, y_coords_val)

        # Reproject destination 2D coordinates in the grid CRS
        x_coords_val, y_coords_val = _reproject_grid_cached(x_coords_val, y_coords_val, dst_coords_crs[2], src_crs)

        return xr.DataArray(x_coords_val, dims=dst_spatial_dims), xr.DataArray(y_coords_val, dims=dst_spatial_dims)

    def _coarsen_interp() -> xr.Dataset:
        """
        Coarsen input dataset if needed (downsampling) and perform regular interpolation
        """

        src_coords_crs = src_spatial_dims + (src_crs,)
        res_ratio_x, res_ratio_y = _get_coarsen_ratio(
            data,
            src_spatial_dims,
            src_coords_crs,
            dst_grid,
            dst_spatial_dims,
            dst_coords_crs,
        )
        interp_data = data
        if (res_ratio_x > 1) or (res_ratio_y > 1):
            coarsen_specs = {
                dim: ratio
                for dim, ratio in (
                    (src_spatial_dims[0], res_ratio_y),
                    (src_spatial_dims[1], res_ratio_x),
                )
                if ratio > 1
            }
            # reduce(np.mean) propagates NaNs like mean(skipna=False), which fails with xarray 2026.9.0
            interp_data = interp_data.coarsen(coarsen_specs, boundary="trim").reduce(np.mean)

        interp_data = _y_increasing(interp_data, src_spatial_dims[0])

        interp_data = interp_data.interp(
            {src_spatial_dims[1]: x_coords, src_spatial_dims[0]: y_coords},
            method=method,
            assume_sorted=True,
        ).drop_vars(src_spatial_dims)
        cast_dtype(interp_data, data)
        interp_data.attrs = _combine_attrs(data.attrs, dst_grid.attrs)
        interp_data = interp_data.assign_coords(dst_grid.coords)
        return interp_data

    data = _drop_coords_reg(data, src_spatial_dims)
    src_crs = dm.get_grid_crs(data)

    if _cond_dst_1d():
        dst_crs = dm.get_grid_crs(dst_grid)
        if src_crs == dst_crs:
            dst_coords_crs = dst_spatial_dims + (dst_crs,)
            x_coords, y_coords = (
                dst_grid[dst_spatial_dims[1]].drop_vars(dst_spatial_dims[1]),
                dst_grid[dst_spatial_dims[0]].drop_vars(dst_spatial_dims[0]),
            )
            return _coarsen_interp()

    dst_grid = dst_grid.transpose(*dst_spatial_dims)
    # chose to use lat/lon if present in dst_grid otherwise native coordinates
    dst_coords_crs = dm.get_coords_name_crs(dst_grid, dst_spatial_dims)

    x_coords, y_coords = _reproject_dst_2d()

    return _coarsen_interp()


def _flatten_concat_grids(  # pylint: disable=too-many-locals
    data: xr.DataArray,
    spatial_dims: tuple[str, ...],
    dst_grid: xr.Dataset,
    dst_spatial_dims: tuple[str, str],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flattens the grids and concatenate columnwise `x` and `y` coordinates into a
    single 2D array. Also, reprojects the destination grid coordinates if they are not
    in the same CRS than the input model coordinates.
    """

    def _choose_crs_meters() -> CRS:
        """
        Choose the target CRS with units in [m] for reprojection.
        """

        if reproj_dst:
            if grid_crs in SWISS_CRS:
                return grid_crs
            return CRS.from_string(SWISS_CRS[1]) if dst_grid_crs == SWISS_CRS[1] else CRS.from_string(SWISS_CRS[0])

        if dst_grid_crs in SWISS_CRS:
            return dst_grid_crs
        return CRS.from_string(SWISS_CRS[1]) if grid_crs == SWISS_CRS[1] else CRS.from_string(SWISS_CRS[0])

    data, grid_y, grid_x, grid_crs = _flatten_coords_crs(data, spatial_dims)
    flat_dim = spatial_dims[0] if len(spatial_dims) == 1 else "cell"
    dst_grid, dst_grid_y, dst_grid_x, dst_grid_crs = _flatten_coords_crs(dst_grid, dst_spatial_dims, flat_dim)

    reproj_dst = dst_grid.sizes[flat_dim] <= data.sizes[flat_dim]

    if method in HEIGHT_METHODS:
        surf_alt = dm.get_surf_alt(data)
        dst_surf_alt = dm.get_surf_alt(dst_grid)
        # include elevation
        coords_flat = np.c_[
            data[grid_x].values,
            data[grid_y].values,
            data[surf_alt].values,
        ]
        dst_coords_flat = np.c_[
            dst_grid[dst_grid_x].values,
            dst_grid[dst_grid_y].values,
            dst_grid[dst_surf_alt].values,
        ]
        target_crs = _choose_crs_meters()
        coords_flat[:, 0], coords_flat[:, 1] = _reproject_grid_cached(
            coords_flat[:, 0], coords_flat[:, 1], grid_crs, target_crs
        )
        dst_coords_flat[:, 0], dst_coords_flat[:, 1] = _reproject_grid_cached(
            dst_coords_flat[:, 0], dst_coords_flat[:, 1], dst_grid_crs, target_crs
        )
        return coords_flat, data.values, dst_coords_flat

    # Non-height methods
    coords_flat = np.c_[data[grid_x].values, data[grid_y].values]
    dst_coords_flat = np.c_[dst_grid[dst_grid_x].values, dst_grid[dst_grid_y].values]
    if reproj_dst:  # reproject destination coordinates in the grid CRS
        dst_coords_flat[:, 0], dst_coords_flat[:, 1] = _reproject_grid_cached(
            dst_coords_flat[:, 0], dst_coords_flat[:, 1], dst_grid_crs, grid_crs
        )
        return coords_flat, data.values, dst_coords_flat

    # reproject grid coordinates in the destination CRS
    coords_flat[:, 0], coords_flat[:, 1] = _reproject_grid_cached(
        coords_flat[:, 0], coords_flat[:, 1], grid_crs, dst_grid_crs
    )
    return coords_flat, data.values, dst_coords_flat


@overload
def _flatten_coords_crs(
    grid: xr.DataArray, spatial_dims: tuple[str, ...], flat_dim: str = "cell"
) -> tuple[xr.DataArray, str, str, CRS]: ...
@overload
def _flatten_coords_crs(
    grid: xr.Dataset, spatial_dims: tuple[str, ...], flat_dim: str = "cell"
) -> tuple[xr.Dataset, str, str, CRS]: ...


def _flatten_coords_crs(
    grid: xr.DataArray | xr.Dataset,
    spatial_dims: tuple[str, ...],
    flat_dim: str = "cell",
) -> tuple[xr.DataArray | xr.Dataset, str, str, CRS]:
    """
    Flatten input dataset if not already and extract spatial coordinates and
    corresponding crs to be used.
    """

    if len(spatial_dims) > 1:
        # flatten grid
        grid = grid.stack({flat_dim: spatial_dims}).transpose(flat_dim, ...)
        spatial_dims = (flat_dim,)

    grid_y, grid_x, grid_crs = dm.get_coords_name_crs(grid, spatial_dims)

    return grid, grid_y, grid_x, grid_crs


def slice_boundingbox(
    data: xr.Dataset,
    poi_info: gpd.GeoDataFrame,
    method: str,
    spatial_dims: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Slice input dataset spatial dimensions to the bounding box of the target POIs
    plus a buffer depending on the interpolation method and corresponding kwargs.
    """

    if spatial_dims is None:
        spatial_dims = dm.get_spatial_dims(data)

    grid_y, grid_x, grid_crs = dm.get_coords_name_crs(data, spatial_dims)
    pois_coords = _convert_poi(poi_info, grid_crs, data[grid_y].dtype)
    yx_bnds = (
        (pois_coords[1].min().item(), pois_coords[1].max().item()),
        (pois_coords[0].min().item(), pois_coords[0].max().item()),
    )

    buffer = _get_buffer(method, **kwargs)
    sub_domain, _ = _get_sub_domain(data, spatial_dims, (grid_y, grid_x), yx_bnds, buffer)
    return data.isel(sub_domain)


def slice_coarsen(  # pylint: disable=too-many-locals
    data: xr.Dataset,
    dst_grid: xr.Dataset,
    method: str,
    spatial_dims: tuple[str, ...] | None = None,
    dst_spatial_dims: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Slice input dataset to the extend of the destination grid and coarsen it if the
    destination grid has lower resolution (downsampling).
    """

    def extra_buffer(ratio: int) -> int:
        """
        Extra buffer to add when input data is coarsened.
        """

        if buffer == 1:
            return ceil(ratio / 2)

        return (ratio - 1) * (buffer - 1)

    if spatial_dims is None:
        spatial_dims = dm.get_spatial_dims(data)
    if dst_spatial_dims is None:
        dst_spatial_dims = dm.get_spatial_dims(dst_grid)

    coords_crs = dm.get_coords_name_crs(data, spatial_dims)
    dst_coords_crs = dm.get_coords_name_crs(dst_grid, dst_spatial_dims)
    grid_crs, dst_grid_crs = coords_crs[2], dst_coords_crs[2]

    x_coords, y_coords = (
        dst_grid[dst_coords_crs[1]].values,
        dst_grid[dst_coords_crs[0]].values,
    )

    if grid_crs != dst_grid_crs:
        if x_coords.ndim == 1:
            x_coords, y_coords = np.meshgrid(x_coords, y_coords)

        # only keep the outer rectangle
        x_coords = np.append(x_coords[[0, -1], :].ravel(), x_coords[1:-1, [0, -1]].ravel())
        y_coords = np.append(y_coords[[0, -1], :].ravel(), y_coords[1:-1, [0, -1]].ravel())

        # Reproject destination coordinates in the grid CRS
        x_coords, y_coords = reproject(x_coords, y_coords, dst_grid_crs, grid_crs)

    yx_bnds = ((y_coords.min(), y_coords.max()), (x_coords.min(), x_coords.max()))

    buffer = _get_buffer(method, **kwargs)
    sub_domain, res = _get_sub_domain(data, spatial_dims, coords_crs[:2], yx_bnds, buffer)
    data_sub = data.isel(sub_domain)

    # coarsen input dataset if needed (downsampling)
    res_ratio_x, res_ratio_y = _get_coarsen_ratio(
        data_sub,
        spatial_dims,
        coords_crs,
        dst_grid,
        dst_spatial_dims,  # type: ignore[arg-type]
        dst_coords_crs,
        res,
    )
    if (res_ratio_x > 1) or (res_ratio_y > 1):
        if len(spatial_dims) == 1:
            logger.warning(
                "Unstructured grid cannot be coarsened. Interpolation to a lower"
                " resolution grid may create aliasing artifacts."
            )
            return data_sub

        coarsen_specs = {
            dim: ratio
            for dim, ratio in (
                (spatial_dims[1], res_ratio_x),
                (spatial_dims[0], res_ratio_y),
            )
            if ratio > 1
        }
        sub_domain.update(
            {
                dim: slice(  # type: ignore[misc]
                    max(sub_domain[dim].start - extra_buffer(ratio), 0),  # type: ignore[union-attr]
                    sub_domain[dim].stop + extra_buffer(ratio),  # type: ignore[union-attr]
                )
                for dim, ratio in coarsen_specs.items()
            }
        )
        # reduce(np.mean) propagates NaNs like mean(skipna=False), which fails with xarray 2026.9.0
        return data.isel(sub_domain).coarsen(coarsen_specs, boundary="trim").reduce(np.mean)

    return data_sub


def _get_buffer(method: str, **kwargs: Any) -> int:
    """
    Buffer specific to the interpolation method to add when slicing input dataset
    to target grid or POIs
    """

    buffer = 1
    if (method == "cubic") or ("3D" in method):
        buffer += 1
    if "k" in kwargs:
        buffer += ceil(np.sqrt(kwargs["k"] / np.pi))
    if "r_search" in kwargs:
        buffer += ceil(kwargs["r_search"])
    return buffer


def _get_sub_domain(  # pylint: disable=too-many-locals
    data: xr.Dataset,
    spatial_dims: tuple[str, ...],
    spatial_coords: tuple[str, str],
    yx_bnds: tuple[tuple[float, float], tuple[float, float]],
    buffer: int,
) -> tuple[dict[str, slice], None] | tuple[dict[str, NDArray], float]:
    """
    Get a dict to index ``data`` in ``spatial_dims`` according to ``yx_bnds``
    plus a ``buffer``. Also return an estimate of the resolution if input grid is
    unstructured.
    """

    def get_cond_xy(buffer_crs: float) -> NDArray[np.bool_]:
        """
        Get a boolean array to indicate which points are within x_bnds and y_bnds
        +- ``buffer_crs`` to index data on the unstructured grid dimension.
        """

        cond_x = (data[grid_x].values > x_bnds[0] - buffer_crs) & (data[grid_x].values < x_bnds[1] + buffer_crs)
        cond_y = (data[grid_y].values > y_bnds[0] - buffer_crs) & (data[grid_y].values < y_bnds[1] + buffer_crs)
        return cond_x & cond_y

    def get_ind_startend(coordname: str, dim: str, bnds: tuple[float, float]) -> tuple[int, int]:
        """
        Get start and end indices to slice data on ``dim`` based on ``bnds``.
        """

        coords = data[coordname]
        dim_size = data.sizes[dim]

        bool_decrease = dm.is_decreasing(coords, dim)

        if bool_decrease:
            coords = coords.isel({dim: slice(None, None, -1)})

        # Find start index: first value greater than lower bound
        id_start = (coords > bnds[0]).argmax(dim).values.min().item()

        # Find end index: first value greater than upper bound, or use full size
        end_mask = coords > bnds[1]
        id_end = end_mask.argmax(dim).values.max().item() if end_mask.values.any() else dim_size

        if bool_decrease:
            start_ori = id_start
            id_start = dim_size - id_end
            id_end = dim_size - start_ori

        return id_start, id_end

    grid_y, grid_x = spatial_coords
    y_bnds, x_bnds = yx_bnds

    if len(spatial_dims) == 1:
        res_start = max(abs(np.diff(data[grid_x][:2])[0]), abs(np.diff(data[grid_y][:2])[0]))
        res_end = max(abs(np.diff(data[grid_x][-2:])[0]), abs(np.diff(data[grid_y][-2:])[0]))
        res_approx = max(res_start, res_end)
        cond_xy = get_cond_xy(2 * res_approx)
        res = _get_flat_grid_res(data[grid_x][cond_xy], data[grid_y][cond_xy])
        cond_xy = get_cond_xy(buffer * res * 1.5)
        return {spatial_dims[0]: cond_xy}, res

    y_dim, x_dim = spatial_dims
    idx_start, idx_end = get_ind_startend(grid_x, x_dim, x_bnds)
    idy_start, idy_end = get_ind_startend(grid_y, y_dim, y_bnds)
    idx_start = max(idx_start - buffer, 0)
    idx_end += buffer
    idy_start = max(idy_start - buffer, 0)
    idy_end += buffer
    return {x_dim: slice(idx_start, idx_end), y_dim: slice(idy_start, idy_end)}, None


def _get_flat_grid_res(coords_x: xr.DataArray, coords_y: xr.DataArray) -> float:
    """
    Get approximative resolution of a flattened grid. Assumes similar resolution
    in both horizontal dimensions.
    """

    def dcoords(coords: xr.DataArray) -> float:
        return np.median(abs(np.diff(coords.values)))

    return max(dcoords(coords_x), dcoords(coords_y))


def _get_coarsen_ratio(  # pylint: disable=too-many-locals
    data: xr.Dataset,
    spatial_dims: tuple[str, ...],
    coords_crs: tuple[str, str, CRS],
    dst_grid: xr.Dataset,
    dst_spatial_dims: tuple[str, str],
    dst_coords_crs: tuple[str, str, CRS],
    res: float | None = None,
) -> tuple[int, int]:
    """
    Get the resolution ratios in `x` and `y` between input data and destination grid
    to eventually coarsen the input data before interpolation.
    """

    if res is not None:
        res_src_x = res_src_y = res
    else:
        center_src_x, center_src_y = _center_coords(data, spatial_dims, coords_crs[:2])  # type: ignore[arg-type]
        res_src_x = center_src_x.diff(spatial_dims[1]).values.mean()
        res_src_y = center_src_y.diff(spatial_dims[0]).values.mean()

    center_dst_x, center_dst_y = _center_coords(dst_grid, dst_spatial_dims, dst_coords_crs[:2])
    src_crs, dst_crs = coords_crs[2], dst_coords_crs[2]
    if src_crs != dst_crs:
        center_dst_x.values, center_dst_y.values = reproject(center_dst_x.values, center_dst_y.values, dst_crs, src_crs)

    res_dst_x = center_dst_x.diff(dst_spatial_dims[1]).values.mean()
    res_dst_y = center_dst_y.diff(dst_spatial_dims[0]).values.mean()
    return abs(round(res_dst_x / res_src_x)), abs(round(res_dst_y / res_src_y))


def _center_coords(
    grid: xr.Dataset, dims: tuple[str, str], coords: tuple[str, str]
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Returns the "x" and "y" coordinates for the 4 center points of input grid domain.
    Used to compute the resolution of the grid.
    """

    y_dim, x_dim = dims
    grid_y, grid_x = coords
    mid_x = int(grid.sizes[x_dim] / 2)
    mid_y = int(grid.sizes[y_dim] / 2)
    sub_x = grid[grid_x].isel({x_dim: slice(mid_x - 1, mid_x + 1)})
    sub_y = grid[grid_y].isel({y_dim: slice(mid_y - 1, mid_y + 1)})
    if y_dim in sub_x.dims:
        sub_x = sub_x.isel({y_dim: slice(mid_y - 1, mid_y + 1)})
        sub_y = sub_y.isel({x_dim: slice(mid_x - 1, mid_x + 1)})
    else:
        sub_y, sub_x = xr.broadcast(sub_y, sub_x)

    return sub_x, sub_y


def _combine_attrs(src_attrs: dict, dst_attrs: dict) -> dict:
    """
    Combine attrs from source and destination grids
    """

    from_dst = ("grid_mapping", "crs", "bounding_box")

    combined_attrs = {key: src_attrs[key] for key in SRC_ATTRS_KEEP if key in src_attrs}
    combined_attrs.update({key: dst_attrs[key] for key in from_dst if key in dst_attrs})
    return combined_attrs


HEIGHT_METHODS = (
    "temp",
    "temp_fixgrad",
    "snowlim",
    "nearest_alt",
    "idw_3D",
    "nearest_3D",
    "linear_3D",
    "upscaled_3D",
)

INTERP_FUNC = {
    "temp": interp_temp,
    "temp_fixgrad": interp_temp_fixgrad,
    "nearest_alt": interp_nearest_alt,
    "snowlim": interp_snowlim,
    "nearest": interp_nearest,
    "nearest_3D": interp_nearest,
    "idw": interp_idw,
    "idw_3D": interp_idw,
    "upscaled": interp_upscaled,
    "upscaled_3D": interp_upscaled,
}
