# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module to wrap some sun position utility functions
"""

import logging
from typing import cast

import numpy as np
from pandas import date_range, Timedelta, offsets
import xarray as xr

from grid_process.data_model import get_validtime

logger = logging.getLogger(__name__)


def zenith(times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray, refrac_cor: bool = True) -> xr.DataArray:
    """
    Analytical expression of solar zenith angle based on spherical
    trigonometry.

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps to be checked in UTC
    lat: xarray.DataArray 1D or 2D
        contains latitudes of the points to be checked
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    refrac_cor: bool
        whether to account for atmospheric refraction. It is significant only when
        the sun is close to horizon, in which case atmospheric refraction correction
        is about 0.6 [degrees]. Default is True

    Returns
    -------
    zen : xarray.DataArray 2D or 3D
        zenith in [degrees] for each time/point couple

    References
    ----------
    * J. A. Duffie and W. A. Beckman,  "Solar Engineering of Thermal
      Processes, 3rd Edition" pp. 14, J. Wiley and Sons, New York (2006)

    * J. H. Seinfeld and S. N. Pandis, "Atmospheric Chemistry and
      Physics" p. 132, J. Wiley (1998)

    * Daryl R. Myers, "Solar Radiation: Practical Modeling for
      Renewable Energy Applications", p. 5 CRC Press (2013)

    * `Wikipedia: Solar Zenith Angle
      <https://en.wikipedia.org/wiki/Solar_zenith_angle>`_

    * `PVCDROM: Sun's Position
      <http://www.pveducation.org/pvcdrom/2-properties-sunlight/
      suns-position>`_

    See Also
    --------
    :func:`azim_elev`, :func:`elevation`
    """

    zen = _zenith(*params_zenith(times, lat, lon)) * (180.0 / np.pi)
    if refrac_cor:
        zen -= _atmo_refraction_corr(90 - zen)

    return zen


def _zenith(decl: xr.DataArray, lat_rad: xr.DataArray, h_angle: xr.DataArray) -> xr.DataArray:
    """
    Compute the formula for the solar zenith angle. The returned angle is in radians
    """

    return np.arccos(np.cos(decl) * np.cos(lat_rad) * np.cos(h_angle) + np.sin(decl) * np.sin(lat_rad))


def cos_zenith(times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """
    Compute the cosine of the solar zenith angle. Atmospheric refraction is NOT
    considered
    """

    decl, lat_rad, h_angle = params_zenith(times, lat, lon)
    return np.cos(decl) * np.cos(lat_rad) * np.cos(h_angle) + np.sin(decl) * np.sin(lat_rad)


def params_zenith(
    times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    get the different parameters to compute zenith angle
    """

    decl = _declination(times.dt.dayofyear)
    h_angle = hour_angle(times, lon)
    lat_rad = lat * (np.pi / 180.0)

    return decl, lat_rad, h_angle


def _declination(dayofyear: xr.DataArray) -> xr.DataArray:
    """
    Solar declination from Duffie & Beckman and attributed to Spencer (1971)
    and Iqbal (1983).
    """

    day_angle = (2.0 * np.pi / 365.0) * (dayofyear - 1)
    return (
        0.006918  # type: ignore[return-value]
        - 0.399912 * np.cos(day_angle)
        + 0.070257 * np.sin(day_angle)
        - 0.006758 * np.cos(2.0 * day_angle)
        + 0.000907 * np.sin(2.0 * day_angle)
        - 0.002697 * np.cos(3.0 * day_angle)
        + 0.00148 * np.sin(3.0 * day_angle)
    )


def _equation_of_time(dayofyear: xr.DataArray) -> xr.DataArray:
    """
    Difference in time between solar time and mean solar time in minutes. From
    .. _PVCDROM: http://www.pveducation.org/pvcdrom/2-properties-sunlight/solar-time
    """

    bday = (2.0 * np.pi / 365.0) * (dayofyear - 81)
    return 9.87 * np.sin(2.0 * bday) - 7.53 * np.cos(bday) - 1.5 * np.sin(bday)  # type: ignore[return-value]


def hour_angle(times: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """
    Hour angle [radians] in local solar time. Zero at local solar noon.

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps in UTC for which to calculate the hour angle
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    """

    hour = (times - times.dt.floor("D")) / np.timedelta64(1, "h")
    eot = _equation_of_time(times.dt.dayofyear)
    return (15.0 * hour + eot / 4.0 + (lon - 180.0)) * (np.pi / 180.0)


def _atmo_refraction_corr(
    elev: xr.DataArray | np.ndarray,
) -> xr.DataArray | np.ndarray:
    """atmospheric refraction correction [degrees]"""

    elev = elev.clip(-1.0, 90.0)
    return (1.02 / np.tan((elev + 10.3 / (elev + 5.11)) * (np.pi / 180.0)) + 0.0019279) / 60


def elevation(times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray, refrac_cor: bool = True) -> xr.DataArray:
    """
    Analytical expression of solar elevation angle based on spherical
    trigonometry

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps to be checked in UTC
    lat: xarray.DataArray 1D or 2D
        contains latitudes of the points to be checked
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    refrac_cor: bool
        whether to account for atmospheric refraction. It is significant only when
        the sun is close to horizon, in which case atmospheric refraction correction
        is about 0.6 [degrees]. Default is True

    Returns
    -------
    elev : xarray.DataArray 2D or 3D
        elevation in [degrees] for each time/point couple

    See Also
    --------
    :func:`azim_elev`, :func:`zenith`
    """

    elev = 90.0 - zenith(times, lat, lon, refrac_cor=False)
    if refrac_cor:
        elev += _atmo_refraction_corr(elev)

    return elev


def azim_elev(
    times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray, refrac_cor: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Analytical expression of solar azimuth and elevation angles based on spherical
    trigonometry

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps to be checked in UTC
    lat: xarray.DataArray 1D or 2D
        contains latitudes of the points to be checked
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    refrac_cor: bool
        whether to account for atmospheric refraction. It is significant only when
        the sun is close to horizon, in which case atmospheric refraction correction
        is about 0.6 [degrees]. Default is True

    Returns
    -------
    azim : xarray.DataArray 2D or 3D
        azimuth in [degrees] for each time/point couple
    elev : xarray.DataArray 2D or 3D
        elevation in [degrees] for each time/point couple

    References
    ----------
    * J. A. Duffie and W. A. Beckman,  "Solar Engineering of Thermal
      Processes, 3rd Edition" pp. 14, J. Wiley and Sons, New York (2006)

    * J. H. Seinfeld and S. N. Pandis, "Atmospheric Chemistry and Physics"
      p. 132, J. Wiley (1998)

    * `Wikipedia: Solar Azimuth Angle
      <https://en.wikipedia.org/wiki/Solar_azimuth_angle>`_

    * `PVCDROM: Azimuth Angle <http://www.pveducation.org/pvcdrom/2-
      properties-sunlight/azimuth-angle>`_

    See Also
    --------
    :func:`elevation`, :func:`zenith`
    """

    decl, lat_rad, h_angle = params_zenith(times, lat, lon)
    zen = _zenith(decl, lat_rad, h_angle)

    numer = np.cos(zen) * np.sin(lat_rad) - np.sin(decl)
    denom = np.sin(zen) * np.cos(lat_rad)

    # cases that would generate new NaN values are safely ignored here
    # since they are dealt with further below
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_azi = numer / denom

    # when zero division occurs, use the limit value of the analytical
    # expression
    cos_azi = cos_azi.where(~np.isclose(denom, 0.0, rtol=0.0, atol=1e-8), 1.0)
    cos_azi = cos_azi.clip(-1.0, 1.0)

    azim = np.sign(h_angle) * np.arccos(cos_azi) * (180.0 / np.pi) + 180.0
    elev = 90 - zen * (180.0 / np.pi)
    if refrac_cor:
        elev += _atmo_refraction_corr(elev)

    return azim, elev


def rise_set(
    times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray, refrac_cor: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Geometric calculation of solar sunrise and sunset times

    .. warning:: The geometric calculation assumes a circular earth orbit with
        the sun as a point source at its center, and neglects the effect of
        atmospheric refraction on zenith. The error depends on location and
        time of year but is of order 10 minutes.

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps to be checked in UTC
    lat: xarray.DataArray 1D or 2D
        contains latitudes of the points to be checked
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    refrac_cor: bool
        whether to account for atmospheric refraction. When true, the sunrise / sunset
        occurs about 4 minutes earlier / later. Default is True

    Returns
    -------
    sunrise : xarray.DataArray 2D or 3D
        sunrise times for each time/point couple in UTC
    sunset : xarray.DataArray 2D or 3D
        sunset times for each time/point couple in UTC

    References
    ----------
    * J. A. Duffie and W. A. Beckman,  "Solar Engineering of Thermal
      Processes, 3rd Edition," J. Wiley and Sons, New York (2006)

    * Frank Vignola et al., "Solar And Infrared Radiation Measurements,"
      CRC Press (2012)
    """

    times_floor, ind_inv = np.unique(times.dt.floor("D").values, return_inverse=True)
    times_floor = xr.DataArray(times_floor, dims=("dayofyear",))
    ind_inv = xr.DataArray(ind_inv, coords=times.coords)
    dayofyear = times_floor.dt.dayofyear

    lat_rad = lat * (np.pi / 180.0)
    decl = _declination(dayofyear)
    eot = _equation_of_time(dayofyear)

    cos_angle = -np.tan(decl) * np.tan(lat_rad)
    if refrac_cor:
        zen_cor = 90.574 * (np.pi / 180.0)  # true zenith when apparent zenith = 0
        cos_angle += np.cos(zen_cor) / (np.cos(decl) * np.cos(lat_rad))

    # numpy ufuncs applied to a DataArray return a DataArray, which numpy type stubs do not know
    sunset_angle = cast(xr.DataArray, np.arccos(cos_angle)) * (180.0 / np.pi)

    sunrise = _angle_to_timedelta(-sunset_angle, lon, eot) + times_floor
    sunset = _angle_to_timedelta(sunset_angle, lon, eot) + times_floor

    return sunrise[{"dayofyear": ind_inv}], sunset[{"dayofyear": ind_inv}]


def _angle_to_timedelta(angle: xr.DataArray, lon: xr.DataArray, eot: xr.DataArray) -> xr.DataArray:
    """converts angles in [degrees] to np.timedelta64"""

    return ((angle - (lon - 180.0) - eot / 4.0) * (3.6e12 / 15.0)).astype(np.dtype("timedelta64[ns]"))


def daytime(times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray, refrac_cor: bool = True) -> xr.DataArray:
    """
    Return a boolean xarray.DataArray specifying whether times are daytime for
    each point defined by the lat and lon arrays. Daytime is defined here as the
    center of the sun being above the non-topographic flat horizon (i.e. sun elevation > 0)

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps to be checked in UTC
    lat: xarray.DataArray 1D or 2D
        contains latitudes of the points to be checked
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    refrac_cor: bool
        whether to account for atmospheric refraction. When true, the sunrise / sunset
        occurs about 4 minutes earlier / later. Default is True

    Returns
    -------
    da_out : xarray.DataArray of bool
        indicates for each time/point couple whether it is daytime

    See also
    --------
    :func:`rise_set`
    """

    day = xr.DataArray(np.unique(times.dt.floor("D").values), dims=("dayofyear",))
    sunrise, sunset = rise_set(day.assign_coords({"dayofyear": day.dt.dayofyear.values}), lat, lon, refrac_cor)

    isday = xr.apply_ufunc(
        lambda x, r, s: np.logical_and(x > r, x < s),
        times.groupby(times.dt.dayofyear),
        sunrise,
        sunset,
    ).drop_vars("dayofyear")
    isday.name = None  # recent xarray versions keep the name of `times`
    return isday


def topo_rise_set(times: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray, horizon: xr.DataArray) -> xr.Dataset:
    """
    Computes topographic sunrise and sunset based on horizon data. Sunrise / sunset
    is defined as the first / last time the sun path cross the horizon line

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps to be checked in UTC. Must be regularly sampled and
        cover full daytime periods
    lat: xarray.DataArray 1D or 2D
        contains latitudes of the points to be checked
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    horizon : xarray.DataArray 2D or 3D
        topographic horizon in [degrees] of the points to be checked, regularly
        sampled along a dimension `azimuth`

    Returns
    -------
    ds : xarray.Dataset 2D or 3D
        contains topographic sunrise and sunset in UTC for each point and day of
        year in input ``times``

    See Also
    --------
    :func:`azim_elev`, :func:`rise_set`, :func:`daytime`

    Note
    ----
    The higher is the time frequency of ``times``, the more accurate will be the
    sunrise and sunset times as it is interpolated between the two time points
    when the sun cross the topographic horizon.
    """

    horizon = horizon.transpose("azimuth", ...)
    azim, elev = azim_elev(times, lat, lon, refrac_cor=True)
    horizon = horizon.sel(azimuth=azim, method="nearest").drop_vars("azimuth")
    shadow_mask = elev > horizon  # False when shadow, True when sun
    ds = xr.Dataset({"elev": elev, "hori": horizon, "shadow": shadow_mask})

    t_dim = get_validtime(times, raise_notfound=True)
    return ds.groupby(f"{t_dim}.dayofyear").map(_topo_suntimes, args=(t_dim,))


def _topo_suntimes(ds: xr.Dataset, t_dim: str) -> xr.Dataset:
    """
    Return a dataset with topographic sunrise and sunset given a input dataset
    with sun elevation, topographic horizon and shadow mask data variables.
    """

    def _suntime(mode: str) -> xr.DataArray:
        """
        Return topographic sunrise/sunset if mode is "rise"/"set" respectively.
        """

        if mode == "rise":
            first_after = ds.shadow.argmax(t_dim)
            first_after = first_after.where(has_sun, 1)
        else:  # sunset
            n_t = ds.sizes[t_dim]
            first_after = n_t - ds.shadow.isel({t_dim: slice(None, None, -1)}).argmax(t_dim)
            first_after = first_after.where(has_sun, n_t - 1)

        indexer = xr.concat([first_after - 1, first_after], t_dim)
        switch = ds[["elev", "hori"]].isel({t_dim: indexer})
        diff = switch.diff(t_dim).isel({t_dim: 0})
        dt = switch[t_dim].diff(t_dim).isel({t_dim: 0})
        switch = switch.isel({t_dim: 0})

        return switch[t_dim] + dt * (switch.hori - switch.elev) / (diff.elev - diff.hori)

    has_sun = ds.shadow.any(t_dim)

    return xr.Dataset({"sunrise": _suntime("rise"), "sunset": _suntime("set")}).where(has_sun)


def topo_shading_cor(
    times: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    horizon: xr.DataArray,
    sampling_freq: str = "10min",
) -> xr.DataArray:
    """
    Computes topographic shading correction factor for variables

    Parameters
    ----------
    times: xarray.DataArray, 1D
        contains timestamps for which correction factor are computed. Must have
        constant time granularity and be in UTC
    lat: xarray.DataArray 1D or 2D
        contains latitudes of the points to be checked
    lon: xarray.DataArray 1D or 2D
        contains longitudes of the points to be checked
    horizon : xarray.DataArray 2D or 3D
        topographic horizon in [degrees] of the points to be checked, regularly
        sampled along a dimension `azimuth`
    sampling_freq : frequency string as accepted by pandas date functionalities
        time frequency to compare sun elevation with topographic horizon. The
        higher the more precise will be the correction factors but at the cost
        of longer computation time. Should be at least higher than the frequency
        of the input ``times`` array

    Returns
    -------
    da_out : xarray.DataArray 2D or 3D
        contains topographic correction factors for each timestamps in input arg
        ``times``

    See Also
    --------
    :func:`azim_elev`, :func:`topo_rise_set`, :func:`daytime`
    """

    t_dim = get_validtime(times, raise_notfound=True)
    out_freq = offsets.Second(times[:2].diff(t_dim).values[0] / Timedelta("1s")).freqstr
    sample_t = date_range(start=times[0].values, end=times[-1].values, freq=sampling_freq)
    sample_t = xr.DataArray(sample_t, coords={t_dim: sample_t})
    azim, elev = azim_elev(sample_t, lat, lon, refrac_cor=True)
    horizon = horizon.sel(azimuth=azim, method="nearest").drop_vars("azimuth")
    shadow_mask = elev > horizon  # False when shadow, True when sun
    return (
        shadow_mask.resample({t_dim: out_freq}, label="right", closed="right").sum(skipna=False)
        * (Timedelta(sampling_freq) / Timedelta(out_freq))
    ).astype(np.float32)
