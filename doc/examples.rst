========
Examples
========

The following notebooks illustrate the spatial interpolation functions of :mod:`grid_process.grid_interp` with open
data from MeteoSwiss, ECMWF, Copernicus, swisstopo and NOAA. Each one compares several interpolation methods visually
and with a runtime benchmark. The first four examples cover Switzerland, the last two the whole globe.

The notebooks are rendered with the outputs of their last execution. Since open forecast data are only kept for a few
days, re-running them retrieves more recent forecasts. To run them, install the extra dependencies with::

    poetry install --with example

The forecast and reanalysis data are downloaded with `earthkit-data <https://earthkit-data.readthedocs.io>`_. To
keep them between sessions, enable its cache once, e.g. in ``~/.config/earthkit/data/config.yaml``::

    cache-policy: user
    user-cache-directory: /path/to/a/large/disk/earthkit-cache

The ERA5-Land and ERA5 examples additionally need a free account on the
`Copernicus Climate Data Store <https://cds.climate.copernicus.eu>`_, with the API token in ``~/.cdsapirc`` and the
licences of the datasets accepted once on their download pages.

All data are open data under the CC BY 4.0 licence (MeteoSwiss, ECMWF, Copernicus Climate Change Service), the
open data terms of swisstopo, or in the public domain (NOAA). The data sources are credited in each notebook.

.. toctree::
   :maxdepth: 1

   examples/01_interp2poi_icon_wind_gusts
   examples/02_interp2poi_ifs_temperature
   examples/03_interp2grid_era5land_precipitation
   examples/04_interp2grid_valais_temperature
   examples/05_interp2poi_gfs_global_temperature
   examples/06_interp2grid_era5_global_temperature
