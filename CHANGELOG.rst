=========
Changelog
=========

Unreleased
----------

**Changes of results** (compared to 1.1.0, see the entries below for details)

* ``interp2grid`` averages a source grid finer than the destination grid per destination cell, instead of per
  window of source grid points aligned on the source grid. The results change most where the windows were offset
  from the destination cells: the 50 m DEM of the seamless weather provider to the seamless 2 km grid
  (``linear``) by 61.6 m RMSE (380 m max), as the windows were 1 km off the cells in x and ``linear`` averaged two
  of them (a 4 x 2 km box); gridefix ICON-CH1 to the seamless 2 km grid by 2 W m-2 (``idw``) to 6 W m-2
  (``nearest_alt``) RMSE for the diffuse radiation, INCA by less
* NaN are ignored in the averaging: a cell is NaN only if all its source values are NaN (e.g. for the snow limit,
  NaN below the orography). Destination cells at the edge of the source are no longer NaN (e.g. 314 cells for the
  DEM above with ``try_reg=False``, a row for ERA5 to a global 1 degree grid)
* unstructured source grids are averaged too: native ICON-CH1 to the seamless 2 km grid changes by 0.25 to 0.5 K
  RMSE for the 2 m temperature, and gets closer to gridefix ICON-CH1 (surface altitude RMSE 108 m → 56 m with
  ``idw``, 91 m → 71 m with ``linear``)
* ``nearest`` and ``linear`` to a coarser destination grid give the mean of the source points of each destination
  cell, whatever ``try_reg``, instead of interpolating between the averaged windows; the other methods interpolate
  from the mean position of the points of each cell. The averaged values are float32 for float32 and 8 or 16 bit
  integer data, float64 otherwise (``try_reg=True`` returned the source dtype, e.g. truncated int16 means), and the
  dimensions are in the order of the source
* ``slice_coarsen`` returns an unstructured dataset (dimension ``cell`` by default, coordinates in the destination
  CRS) when it averages the source grid
* the slicing buffer of ``interp2grid``, ``slice_coarsen`` and ``slice_boundingbox`` counts grid steps of the
  coarser of the source and destination grids, and the grid step of unstructured sources is estimated from the
  area per point (more robust, e.g. for ICON triangles)
* ``ens_stats.neighbor_stat`` with ``mean`` and NaN: the mean of the valid values over the whole space x time
  window, instead of the mean of the means along each dimension. Without NaN, and for ``sum``, ``min`` and ``max``,
  the results are unchanged (``min``/``max`` identical, ``sum``/``mean`` within float32 rounding). The result is
  float32 for 8 or 16 bit integer and boolean data (``mean``/``sum`` were float64, ``max`` of booleans too)

**Changes**

* changed ``interp2poi`` and ``interp2grid`` to no longer copy a dataset with a single data variable before
  interpolation (new ``copy`` parameter of ``ds_utils.ds_to_da``, default ``True``). For the operational grids
  (with cached neighbors), POIs are interpolated up to about 1.5 times faster (e.g. ICON-CH2 ``nearest`` 0.18 s →
  0.12 s) with 0.1 to 0.2 GB less memory for ICON, and IFS to the seamless 2 km grid with ``nearest_alt`` 0.39 s →
  0.24 s. A sliced source that is not contiguous in memory is now copied into contiguous rows per grid point, which
  is slower for large sources: ICON-CH2 in LV95 to the seamless 2 km grid with ``nearest_alt`` or ``temp_fixgrad``
  0.23 s → 0.29 s and 0.34 s → 0.39 s
* changed the averaging of a source grid finer than the destination grid by ``interp2grid`` (and
  ``slice_coarsen``): the mean of the source points of each destination cell (the cell of their nearest destination
  node along x and y), with a sparse matrix product, for regular and unstructured sources (unstructured sources
  were not averaged, with a warning), if the destination grid has regular 1D coordinates in its CRS (otherwise
  with a warning). For the seamless 2 km grid (with cached neighbors): gridefix ICON-CH1 ``idw`` 0.43 s → 0.27 s,
  ``nearest_alt`` 0.31 s → 0.16 s, ``linear`` 0.54 s → 0.13 s, with up to 0.13 GB more peak memory (copy of the
  sliced source into contiguous rows per grid point, e.g. ``nearest_alt`` 0.07 GB → 0.20 GB); the 50 m DEM
  ``linear`` 0.13 s → 0.07 s; native ICON-CH1 ``linear`` 0.03 s instead of 1.05 s without averaging.
  ``interp2grid`` on the output of ``slice_coarsen`` (with the same destination grid and method) gives the same
  result as on the source grid
* added ``slice_grid``: slices a source grid to a destination grid, keeping its structure and CRS (the grid
  counterpart of ``slice_boundingbox``)
* added the support of unstructured grids to ``ens_stats.neighbor_stat`` and ``upscaled_prob``: the neighborhood
  of each point is its ``w_size_xy**2`` nearest points (the window of a square grid for 3 and 5), with the nearest
  neighbors searched in meters and cached, ignoring NaN. Native ICON-CH1 (34 fields, 1.15 M points, window 5):
  ``mean``/``sum`` 0.5 s, ``min``/``max`` 0.8 s, plus 1.8 s for the first neighbor search (``rolling`` on a
  regular grid of the same size: 2.4 to 4.3 s)
* changed ``ens_stats.neighbor_stat`` (and ``upscaled_prob``) on regular grids and along time to
  ``scipy.ndimage`` filters, applied to all the dimensions of the window at once (dask: one chunk along them).
  ICON-CH1-EPS gridefix (34 x 11 x 252 x 376) ``max`` with windows 17 and 3 (data4web-pipelines): 3.8 s → 1.1 s,
  peak memory 1.07 GB → 0.32 GB; a regular grid of 34 x 1072 x 1072 with window 5: ``mean`` 4.3 s → 0.6 s, ``sum``
  2.4 s → 0.5 s, ``max`` 3.6 s → 1.0 s; native ICON-CH1 with windows 5 and 3: ``mean``/``max`` 3.4 s → 1.1 s
* changed the slicing of unstructured grids opened lazily from a file (e.g. ``xarray.open_dataset`` of a netCDF
  file, without dask) by ``interp2poi``, ``interp2grid``, ``slice_boundingbox``, ``slice_grid`` and
  ``slice_coarsen``: the variables on the unstructured dimension are read by chunks of 16 MB, each with one slice
  (only the chunks with selected points), and loaded, instead of reading the selected points one by one. Native
  ICON-CH1 netCDF to the seamless 2 km grid: ``slice_grid`` and loading 50 s → 1.4 s. Sources in memory, dask
  arrays and regular grids are unchanged
* changed ``interp2poi`` and ``interp2grid`` with ``upscaled``: the statistic is computed by blocks of POIs, without
  gathering the ``k`` neighbors of all the POIs at once, and with ``idw``: a product with the sparse matrix of the
  weights, same results. Native ICON-CH1-EPS (10 members x 34 steps) to the 5,622 production POIs of
  data4web-pipelines with ``upscaled`` and ``k=289``: 1.19 s → 0.39 s, peak memory +2.5 GB → +0.3 GB (first call
  1.41 s → 0.55 s); gridefix ICON-CH1-EPS (3 variables) to the seamless 2 km grid with ``idw``: 0.78 s → 0.43 s,
  peak memory +0.92 GB → +0.31 GB
* changed ``nn_lookup_cached``: new ``distances`` parameter (in the cache key), the distances are None if
  ``False``, and the indices are int32: the cache of the methods without distances (``nearest``, ``upscaled``,
  ``neighbor_stat`` on unstructured grids) takes 4 times less memory (native ICON-CH1 ``neighbor_stat`` with
  ``w_size_xy=17``: 1.3 GB instead of 5.3 GB)
* added ``values`` to the default names of the unstructured grid dimension
  (``GRID_PROCESS__data_model__grid__unstructured_dim``): the dimension of non-regular GRIB grids read with cfgrib or
  earthkit-data, e.g. ICON open data, no longer needs to be renamed

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
