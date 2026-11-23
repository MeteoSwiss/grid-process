# AGENTS.md

Guidance for coding agents (and humans) working on grid-process. User documentation is in `doc/` and
<https://meteoswiss.github.io/grid-process/>. Contribution rules are in `CONTRIBUTING.md`.

## What this library is

grid-process processes weather forecast `xarray` datasets. Its main part is spatial interpolation
(`grid_process/grid_interp.py`): from regular or unstructured source grids to points of interest (POIs,
`interp2poi`) or to another grid (`interp2grid`). It also has time series, aggregation, ensemble, blending, units and
sun utilities.

**It runs operationally at MeteoSwiss** on the Swiss domain, every forecast run, for every model and variable:

- the data4web pipelines call `interp2poi(..., try_reg=False)` on forecast and nowcasting fields to the production
  POIs (about 5 600), mostly with `idw`, `nearest_alt`, `nearest`, `temp` and `snowlim`;
- the seamless weather provider calls `interp2grid(..., try_reg=False)` (and `slice_coarsen`) on forecast fields to a
  Swiss 2 km LV95 grid, mostly with `nearest_alt`, `idw`, `linear` and `temp_fixgrad`;
- both interpolate static fields (DEM, topographic descriptors, sky view factor, horizon) from 25 m and 50 m LV95
  grids opened lazily from zarr (dask), with `linear` or `nearest` (int8 horizon) and the default `try_reg=True`.

Both applications may switch from the ICON-CH1/CH2 fields of the grid database (structured rotated grids) to native
ICON (unstructured, most likely a Swiss cut-out from polytope) for all their uses: the unstructured path must handle
every operational case.

So:

- **Results are a contract.** Any change of the numbers must be intended, measured on the operational cases (below)
  and documented in `CHANGELOG.rst` under **Changes of results**, with its size (e.g. RMSE, max difference).
- **Runtime and memory matter, with cold caches first.** The applications run as cron jobs (OpenShift) whose memory
  is not kept between runs, so most interpolations start with empty caches. Within a job, several interpolations
  reuse the warm caches when the grids, POIs and kwargs are the same. Measure both the first and the repeated call,
  and the peak memory.
- **Never assume, measure.** Check a performance or behaviour claim on data of operational size before stating it.
- As equivalent features become available in ECMWF earthkit, parts of this library may be deprecated in its favour
  (see `CONTRIBUTING.md`).

## Commands

```console
poetry install                        # Python 3.11 to 3.14
poetry run pytest                     # CI: pytest -s --cov=grid_process --cov-fail-under=80
poetry run ruff format --check        # line length 120
poetry run ruff check
poetry run mypy grid_process          # all functions are typed (disallow_untyped_defs)
poetry run pylint grid_process        # keep 10.00/10
poetry run sphinx-build doc doc/_build
```

## Layout

- `grid_process/grid_interp.py`: spatial interpolation (`interp2poi`, `interp2grid`, `slice_grid`,
  `slice_coarsen`, `slice_boundingbox`, the `interp_<method>` functions, nearest neighbour lookups and caches).
- `grid_process/grid_interp_utils.py`: creation of destination grids.
- `grid_process/data_model.py`: finds the names of dimensions and coordinates (spatial, temporal, ensemble, surface
  altitude) and the CRS of a dataset. Every module goes through it rather than hard-coding names.
- `grid_process/config/`: `grid_process.yaml` (supported names, cache sizes, `lv95_max_offset`) and its pydantic
  settings. Everything can be overridden with `GRID_PROCESS__<section>__<key>` environment variables, read when
  `grid_process` is imported.
- `grid_process/ens_stats.py`: ensemble probabilities, `neighbor_stat` and `upscaled_prob` (regular and unstructured).
- `grid_process/_sparse.py`: sparse matrix helpers shared by `grid_interp` and `ens_stats` (package internal).
- `grid_process/optimization.py`: `memoize`, the in-memory LRU cache used for nearest neighbours and reprojections.
- Other modules: `timeseries`, `aggreg`, `blending`, `ds_utils`, `da_utils`, `units_conv`, `sun`, `time_utils`,
  `helpers` (also `reproject`). Many public functions are also `xarray` accessors (`ds.interp2poi(...)`).
- `test/`: one test module per module; `test/conftest.py` builds small synthetic grids (regular lat/lon, rotated,
  LV95, unstructured) and POIs.
- `doc/examples/*.ipynb`: interpolation examples on open data, with runtime comparisons of the methods (examples,
  not the operational cases, see below).
- `benchmarks/operational_cases.py`: benchmark of the operational cases (see below), smoke-tested by
  `test/test_benchmarks.py`.

## How the interpolation works

`interp2poi(data, poi_info, method, try_reg=True, **kwargs)`:

1. `dm.get_spatial_dims` and `dm.set_grid_mapping` identify the grid and its CRS: a `crs` attribute, the internal
   `grid_mapping` JSON attribute, a CF grid mapping variable, or WGS84 for latitude/longitude names.
2. With `try_reg` and `nearest`/`linear` on a structured grid with 1D coordinates: `xarray` interpolation in the grid
   CRS (`_interp_regular`). The operational forecast fields use `try_reg=False` (the generic path below); the large
   static fields (25 m and 50 m dask grids) use this regular path.
3. Otherwise: slice to the POIs plus a method-dependent buffer (`slice_boundingbox`, `get_buffer`), view the dataset
   as one array with the spatial dimensions first (`ds_to_da(copy=False)`), and flatten it into points
   (`_flatten_concat`). Points that are NaN in all fields are dropped (`_valid_points`, except for `snowlim`), which
   allows masking (`ds.where(land).interp2poi(...)`).
4. Coordinates are in meters, never degrees (`_choose_kdtree_crs`): the native CRS if it is projected in meters
   (e.g. LV95), else the metric CRS of the other side, else LV95 near Switzerland (`lv95_max_offset`), else ECEF
   (global grids; no triangulation methods). Methods with altitude add it as a third coordinate.
5. `INTERP_FUNC[method]` (`interp_idw`, `interp_nearest_alt`, `interp_temp`, ...) or `scipy.interpolate.griddata`
   (`linear`, `linear_3D`, `cubic`) computes the values, which are put back into a dataset with a `poi` dimension.

`interp2grid(data, dst_grid, method, try_reg=True, **kwargs)` follows the same steps with a destination grid
(structured, 2 spatial dimensions). Before them, it slices the source to the destination (`slice_grid`). If the
source is finer (at least 2.25 points per destination cell, `COARSEN_MIN_DENSITY`), it averages it per destination
cell with a sparse 0/1 matrix, ignoring NaN (`slice_coarsen`, `_sparse_mean_ds`). For `nearest` and `linear`, those
cell means are the result. The other methods interpolate from the mean position of the points of each cell.
Unlike `interp2poi`, `interp2grid` keeps the points that are NaN in all fields (`_flatten_concat_grids`): a NaN
source point can give NaN at the destination. Only the averaging per destination cell ignores NaN (a cell is NaN only
if all its source points are).

Things to keep in mind:

- **Caches.** `nn_lookup_cached`, `_nn_lookup_regular_cached`, `_reproject_poi`, `_reproject_grid` and `_to_ecef`
  are memoized on the content of the coordinates (`_array_key`). Their sizes are in the config (`cache_size`, 0
  disables). A cached array is returned to every caller: never modify it in place.
- **dtypes.** float32 data stays float32 (`cast_dtype`), and indices are int32 when possible. Avoid silent
  promotions to float64; they double the memory of the large arrays.
- **Lazy and dask sources.** Regular grids are sliced with slices; lazily opened unstructured sources are read by
  chunks (`_isel_sub_domain`) because point-wise indexing of a netCDF file is very slow.
- **Copies.** A sliced source is often not contiguous. Count the copies of the data in a change: the peak memory of
  an operational call is a few copies of the sliced source.

## Operational cases (benchmark on data of these shapes)

The operational data (internal grid database, native ICON files) is not public. `benchmarks/operational_cases.py`
builds synthetic data with the same structure, CRS, sizes, dtypes and NaN patterns, and runs the operational calls
on it (see below). Shapes are y × x unless stated.

| Case | Grid | Points | Fields |
|---|---|---|---|
| IFS-ENS (grid database) | regular lat/lon, 0.1° | 59 × 168 | 51 members × 145 steps, float32 |
| ICON-CH1-EPS (grid database) | rotated lat/lon (pole 43°N, 190°E), 0.01° | 252 × 376 | 11 members × 34 steps, 3 variables, float32 (radiation all NaN at the first step) |
| ICON-CH1-EPS snow limit (grid database) | rotated lat/lon, 0.01° | 252 × 376 | 11 members × 34 steps, snow limit and freezing level, float32 with NaN (see below) |
| ICON-CH2-EPS (grid database) | rotated lat/lon, 0.02° | 127 × 188 | 21 members × 121 steps, 3 variables, float32 |
| ICON-CH2-EPS lake-masked temperature (grid database) | rotated lat/lon, 0.02° | 127 × 188 | 21 members × 121 steps, float32, lake points all NaN (2 %) |
| ICON-CH2-EPS LV95 (grid database) | LV95 (EPSG:2056), 2 km | 320 × 355 | 21 members × 121 steps, float32 |
| INCA (grid database, analysis and nowcasting) | LV95, 1 km | 640 × 710 | 18 reference times × 1 step, 3 variables, float32 |
| ICON-CH1-EPS native, Swiss cut-out (possible future) | unstructured lat/lon, dimension `values` | about 113 000 | as the grid database ICON-CH1-EPS, `HSURF` (radiation without NaN, as the open data) |
| ICON-CH2-EPS native, Swiss cut-out (possible future) | unstructured lat/lon, dimension `values` | about 28 200 | as the grid database ICON-CH2-EPS, `HSURF` |
| ICON-CH1-EPS native, whole domain (upper bound, not operational) | unstructured lat/lon, dimension `values` | 1 147 980 | 11 members × 34 steps, float32 |
| ICON-CH2-EPS native, whole domain (upper bound, not operational) | unstructured lat/lon, dimension `values` | 283 876 | 21 members × 121 steps, float32 |
| Static fields (DEM, topographic descriptors) | LV95, 25 m and 50 m, zarr (dask) | 10 121 × 15 601 (25 m), 12 800 × 14 200 (50 m) | float32 |
| Horizon (data4web) | LV95, 25 m, zarr (dask, chunks 500 × 500 × 45) | 10 121 × 15 601 | int8 × 360 azimuths (270 used) |
| Seamless destination grid | LV95, 2 km | 125 × 190 | surface altitude (`dem`) |
| Production POIs | EPSG:4326 points with altitude | 5 622 | |

The grid database files carry `latitude`/`longitude`, `surface_altitude` and a `grid_mapping` JSON attribute
(e.g. `{"grid_mapping_name": "rotated_latitude_longitude", "grid_north_pole_latitude": 43.0,
"grid_north_pole_longitude": 190.0, "earth_radius": 6371229.0}`).

The native cut-out cases are the counterparts of the grid database ICON cases (same members, steps, variables,
methods and NaN patterns), on the native points within the area of the grid database grids. The whole domain cases
(one ICON-CH1, one ICON-CH2) are upper bounds, not operational uses: they show the cost of slicing a whole native
domain, if the applications ever got it.

Performance-critical case: **the horizon to the production POIs** (data4web `process_horizon`) used to be one of the
longest operational interpolations. Keep its runtime and memory:

```python
hori = hori_topo.isel(azimuth=slice(45, -45)).grid_interp.interp2poi(poi_def, method="nearest").astype(np.float32)
```

It goes through the regular grid path with dask (`_interp_regular`, `_interp_nearest_regular`, cached nearest grid
indices), and must stay lazy and int8 until the selection of the POIs.

NaN cases to keep correct:

- **Snow limit.** NaN where the snow limit is below the orography: 31 % of the values, and 24 % of the points are
  NaN for all members and steps. data4web interpolates it with `snowlim` (`interp2poi`). The seamless weather
  provider will regrid it to the 2 km grid: the averaging of ICON-CH1 per destination cell must give NaN only where
  all the source points of the cell are NaN.
- **Masked points.** `interp2poi` (outside of the regular grid path) excludes the points that are NaN in all
  fields. data4web masks lakes before interpolating to the POIs, `ds.where(ds.land_area_fraction >= 0.5)
  .drop_vars("land_area_fraction")`, so that a POI near a lake takes its value from land points only.
  `interp2grid` keeps such points (NaN propagates to the destination), so masking before it would not have this
  effect; the seamless weather provider does not mask. This difference is intended, do not align it: POIs are on
  the ground, sometimes near a shore, where lake values (with different properties) would be wrong, whereas lakes
  are still in a destination grid and keep the lake values.

Benchmark a change that can affect runtime, memory or results against `main`, on the same machine (otherwise idle),
with `benchmarks/operational_cases.py`. It runs the operational calls (data4web to the POIs, seamless weather
provider to the 2 km grid, static fields) with their operational settings, each case in fresh processes: runtime of
the first call (cold caches) and of the repeated calls (warm caches), peak memory, and checksums of the outputs.

```console
git worktree add ../grid-process-main main
poetry run python benchmarks/operational_cases.py list
poetry run python benchmarks/operational_cases.py run --grid-process ../grid-process-main --out /tmp/bench/main
poetry run python benchmarks/operational_cases.py run --out /tmp/bench/branch
poetry run python benchmarks/operational_cases.py compare /tmp/bench/main /tmp/bench/branch
```

Select cases with `--case` (shell patterns), keep the outputs to quantify their differences with `--save-outputs`,
and use the real data with `--data-dir` when available. Report the identical outputs, or quantify the differences in
the CHANGELOG. A full run takes about 12 minutes. Keep the outputs outside the repository. Add a case when a new
operational use appears.

`run` never downloads: it uses the synthetic data or the files of `--data-dir`, so that both sides of a comparison
use the same data. Real native ICON comes from the open data of MeteoSwiss: `fetch-native --out DIR` downloads a
run (all members and lead times; runs are available for about a day; needs the network and
`poetry install --with example`) once, then `run --data-dir DIR` uses it. A native case on a real file with fewer
members or lead times than the operational ones is noted in the results.

The example notebooks (`doc/examples/*.ipynb`) also time the methods (first and repeated calls), on open data
(ICON-CH2 native, IFS, ERA5, GFS) and with other grids and settings. They are examples for users, not the
operational cases: do not use them as the reference for a performance change, but rerun the affected notebooks when
a change alters their results or timings noticeably.

## Conventions

- numpy-style docstrings, type hints everywhere, lines up to 120 characters.
- Names, CRS and attributes come from `data_model` and the config, never hard-coded.
- Do not import private (`_name`) functions across modules: shared internals go into a private module with public
  names (e.g. `_sparse.py`).
- New files start with the copyright and SPDX license header of the existing ones.
- `CHANGELOG.rst`: one entry per change under **Unreleased**, with timings on the operational cases for performance
  changes. Changes of results are listed separately under **Changes of results**.
- PR descriptions follow `.github/pull_request_template.md`.

## Releases

1. A release PR renames **Unreleased** in `CHANGELOG.rst` to `X.Y.Z (YYYY-MM-DD)` and adds the version to
   `doc/_static/switcher_config.json` (marked stable).
2. After merging, pushing the tag `X.Y.Z` runs `.github/workflows/CI_publish.yaml`: tests, then build with the tag
   as version (`pyproject.toml` keeps `0.0.0`), PyPI publication and versioned documentation.
