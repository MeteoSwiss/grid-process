=========
Changelog
=========

Unreleased
----------

**Changes of results** (compared to 1.1.0, see the entries below for details)

* the coarsening of the source grid by ``interp2grid`` (to a coarser destination grid) skips NaN: a coarsened
  cell with some NaN values gets the mean of its valid values instead of NaN, only cells with all values NaN stay NaN
* coarsened float32 values differ from before by float32 rounding, growing with the window size (for 2 m
  temperature, up to 3e-5 K for a 2 x 2 window and 2e-4 K for 10 x 10)

**Changes**

* changed ``interp2poi`` and ``interp2grid`` to no longer copy a dataset with a single data variable before
  interpolation (new ``copy`` parameter of ``ds_utils.ds_to_da``, default ``True``). For the operational grids
  (with cached neighbors), POIs are interpolated up to about 1.5 times faster (e.g. ICON-CH2 ``nearest`` 0.18 s →
  0.12 s) with 0.1 to 0.2 GB less memory for ICON, and IFS to the seamless 2 km grid with ``nearest_alt`` 0.39 s →
  0.24 s. A sliced source that is not contiguous in memory is now copied into contiguous rows per grid point, which
  is slower for large sources: ICON-CH2 in LV95 to the seamless 2 km grid with ``nearest_alt`` or ``temp_fixgrad``
  0.23 s → 0.29 s and 0.34 s → 0.39 s
* changed the coarsening of the source grid by ``interp2grid`` to skip NaN (e.g. for the snow limit, NaN below the
  orography), with bottleneck. ICON-CH1 to the seamless 2 km grid (with cached neighbors) is about 1.4 times
  faster (e.g. ``nearest_alt`` 0.30 s → 0.21 s), with about 0.05 GB more peak memory (copy of the windows)

1.1.0 (2026-11-12)
------------------

**Changes of results** (compared to 1.0.0, see the entries below for details)

* ``interp2poi`` and ``interp2grid`` (outside of the regular grid interpolation) search nearest neighbors and
  triangulate in meters instead of latitude/longitude degrees: neighbors, IDW weights and ``linear``/``cubic``
  results change slightly for grids that were searched in degrees, including native LV95 grids with
  latitude/longitude coordinates
* grids in rotated pole or latitude/longitude on a sphere without datum (e.g. a CF grid mapping with
  ``earth_radius``) now get the datum shift to Swiss coordinates (about 180 m)
* wrong values or NaN are fixed for grids with integer coordinates and for latitude/longitude grids whose longitude
  convention (0..360 or -180..180) differs from the one of the POIs or destination grid
* ``create_dst_grid`` (with ``topright`` and ``resolution``) and ``create_dst_grid_from_src`` return different
  destination grids: exact resolution, and the extent of the source domain reprojected in the destination CRS
* ``grid_interp._get_buffer`` is renamed to ``get_buffer`` (now public)

**Changes**

* added a ``py.typed`` marker so that type checkers use the inline type hints of ``grid-process`` (PEP 561)
* added example notebooks to the documentation illustrating ``interp2poi`` and ``interp2grid`` with open data, with
  their dependencies in the optional ``example`` dependency group. Two of them are global: GFS 2 m temperature to
  stations worldwide (compared with GHCN-hourly observations) and ERA5 2 m temperature to the IFS 0.1° grid
* fixed ``interp2grid`` and ``interp2poi`` truncating reprojected coordinates when the grid coordinates are
  integers (e.g. an LV95 destination grid built with ``np.arange``), which gave wrong values or NaN
* fixed ``create_dst_grid`` with ``topright`` and ``resolution``: the resolution is now exact and ``topright``
  a closed upper bound, instead of the grid spacing being stretched to end exactly on ``topright``
* fixed ``interp2poi`` and ``interp2grid`` (outside of the regular grid interpolation) searching nearest neighbors
  and triangulating (``linear``, ``cubic``) in latitude/longitude degrees, where a degree of longitude is shorter
  than a degree of latitude (about 0.68 times in Switzerland). Distances are now always in meters: native
  coordinates projected in meters are used even if latitude/longitude coordinates are present (e.g. LV95 grids),
  and latitude/longitude grids are converted to Swiss LV95, or to earth-centered earth-fixed coordinates (ECEF) far
  from Switzerland. Neighbors, IDW weights and ``linear``/``cubic`` results change slightly for grids that were
  searched in latitude/longitude, including native LV95 grids with latitude/longitude coordinates
* fixed the reprojection of grids in rotated pole (or latitude/longitude) on a sphere without datum, e.g. from a CF
  grid mapping with ``earth_radius``: their latitudes and longitudes are now considered as WGS84 ones, as their
  latitude/longitude coordinates usually are. The datum shift to Swiss coordinates (about 180 m) was left out.
  ``parse_grid_mapping`` (and ``get_grid_crs``) now return such CF mappings directly on WGS84, without
  ``pyproj.CRS.from_cf``, which took about 0.3 s for a rotated pole; other mappings parsed by ``from_cf`` are cached
* added the support of global and other non-Swiss latitude/longitude grids by ``interp2poi`` and ``interp2grid``
  (except ``linear``, ``linear_3D`` and ``cubic``, which raise an error), with nearest neighbor searches in ECEF
  coordinates. The maximum distance of the destination to the LV95 central northing for using LV95 is set by the
  new setting ``lv95_max_offset`` (environment variable ``GRID_PROCESS__lv95_max_offset``, default 700 km)
* changed the methods with altitude, which no longer reproject to Swiss coordinates when the source grid or the
  destination has another CRS projected in meters (e.g. UTM)
* added an explicit error when the destination grid of ``interp2grid`` is unstructured, which is not supported
* fixed ``interp2poi`` and ``interp2grid`` with latitude/longitude source grids whose longitude convention
  (0..360 or -180..180) differs from the one of the POIs or destination grid: the source was sliced to the
  wrong longitudes (wrong values in the generic path, e.g. for a global destination grid) and the regular grid
  interpolation (``try_reg=True``) returned NaN. Global latitude/longitude grids are also interpolated across
  their 360 degree seam. When slicing a latitude/longitude source grid, the longitude buffer is widened by
  1 / cos(latitude) (as nearest neighbors are now searched in meters) and all longitudes are kept close to the
  poles
* changed the ``temp`` method of ``interp2poi``: POIs whose nearest grid points all have the same altitude (e.g.
  close to the poles of a latitude/longitude grid, whose pole rows repeat the same point, or on flat orography)
  use the fixed lapse rate fallback (as POIs far above or below their neighborhood) instead of raising an error
* fixed ``create_dst_grid_from_src``: the destination grid now spans exactly the bounding box of the source domain
  reprojected in the destination CRS (from native coordinates projected in meters when available), instead of the
  reprojected corners of its latitude/longitude bounding box, which could be much larger (e.g. +80 km for ICON-CH2
  in LV95) or miss parts of the domain (e.g. 28 km for IFS in LV95)
* fixed ``set_grid_mapping`` with CF grid mappings holding numpy arrays or non-float64 numpy values (e.g. a
  ``standard_parallel`` array read from a netCDF file), which raised a ``TypeError``
* changed ``memoize``: the least recently used result is evicted (instead of the oldest one), the cache key can be
  computed by an optional ``key`` function (only when caching is active), a cache size of 0 deactivates caching
  without overhead, and the cache is safe to use from several threads, also when the process forks (the fork waits
  until no thread is using the cache). The ``hkey`` keyword argument still works
* fixed the in-memory caches of the POI and grid reprojections, whose keys left out the source CRS, and the cache
  of the regular grid nearest interpolation, which failed with a POI dimension name other than ``poi``
  (``GRID_PROCESS__data_model__poi__dim``)
* documented the in-memory caches of ``interp2poi`` and ``interp2grid`` and how to set their sizes or deactivate
  them with the environment variables ``GRID_PROCESS__cache_size__<name>`` (Usage page)
* changed ``interp2poi`` and ``interp2grid`` outside of the regular grid interpolation to flatten structured
  grids with numpy instead of ``xarray.stack``, and ``interp2poi`` to look for the grid points with NaN in all
  fields (excluded, as before) only among those with NaN in the middle field. Later calls with cached neighbors are
  faster on large grids (e.g. 0.13 s to 0.05 s for a global 0.25 degree grid to 33 POIs, 1.6 s to 0.5 s from this
  grid to a global 0.1 degree grid), with identical results
* fixed ``interp2poi`` raising an ``IndexError`` when all POIs are outside an unstructured grid (no grid point left
  after slicing): they now get NaN, as outside of the regular grid interpolation
* made ``grid_interp.get_buffer`` public (renamed from ``_get_buffer``): the slicing buffer in grid points of an
  interpolation method and its arguments, e.g. to slice the input data with ``slice_boundingbox`` beforehand

1.0.0
-----

First open-source release of ``grid-process``, the continuation of the internal MeteoSwiss library
``gridefix-process`` (last internal version: 6.1.0).

**Breaking Change** (compared to ``gridefix-process`` 6.1.0)

* renamed the package from ``gridefix-process`` to ``grid-process`` (import name ``grid_process``)
* renamed the environment variable prefix from ``GRIDEFIX_PROCESS__`` to ``GRID_PROCESS__``
* versioning restarts at ``1.0.0``
