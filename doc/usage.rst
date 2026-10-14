=====
Usage
=====

To use a module from grid-process in a project::

    import grid_process.<module_name>


Manipulating data
-----------------

Use case: remove 'forecast_reference_time' and 'lead_time' dimensions and set 'time'
dimension with numpy.datetime64 coordinate labels::

    import grid_process.timeseries as tms
    ds_validtime = tms.setindex_validtime(ds)

    # same but with accessor:
    ds_validtime = ds.timeseries.setindex_validtime()

Returns a xarray.Dataset with dimension 'forecast_reference_time' squeezed and
dimension 'time' as coordinates of dtype numpy.datetime64. Also works when the
initial dataset has dimension 'forecast_reference_time' and 'lead_time' of length
greater than 1. In such cases, same validtimes are not duplicated but the data from
the most recent run is used.

Use case: temporal interpolation::

    ds_1hr = tms.interp2granularity(ds, '1h', method='cubic')

    # same but with accessor:
    ds_10min = ds.interp2granularity('10min', method='back_cum')

Returns a xarray.Dataset interpolated to the specified granularity in the 'lead_time'
or 'time' dimension. There are many interpolation methods available for specific
variables such as 'back_cst' (e.g. for mean rainfall rate over last timestamp),
'back_cum' (e.g. for precipitation amount accumulated since previous timestamp),
'sundur' for 'duration of sunshine', 'linear' (e.g. for windspeed, windgust),
'nearest' (e.g. for wind direction), 'cubic' (e.g. for temperature)

Use case: POI interpolation::

    import geopandas as gpd
    import grid_process.grid_interp as gri

    # geopandas.GeoDataFrame with POI metadata (lat, lon, altitude, type, name)
    poi_def = gpd.read_file("poi.geojson")

    ds_poi = gri.interp2poi(ds, poi_def, method='linear')

    # same but with accessor:
    ds_poi = ds.interp2poi(poi_def, method='temp', k=50, z_fact=15)

Returns a xarray.Dataset with dimension 'x' and 'y' replaced by 'poi'. There
are many interpolation methods available for specific variables such as
temperature, snowlimit etc. There are also methods to perform local smoothing,
or taking local extremas.

Use case: regridding::

    import grid_process.grid_interp as gri

    ds_regridded = gri.interp2grid(ds, ds_dst, method='linear')

    # same but with accessor:
    ds_regridded = ds.interp2grid(ds_dst, method='temp_fixgrad', r_search=1.7)

Returns a xarray.Dataset with the data of ds remapped to the coordinates of ds_dst. Same
interpolation methods exist as for POI interpolation
