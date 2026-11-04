=========
Changelog
=========

Unreleased
----------

* added a ``py.typed`` marker so that type checkers use the inline type hints of ``grid-process`` (PEP 561)
* added example notebooks to the documentation illustrating ``interp2poi`` and ``interp2grid`` with open data, with
  their dependencies in the optional ``example`` dependency group
* fixed ``interp2grid`` and ``interp2poi`` truncating reprojected coordinates when the grid coordinates are
  integers (e.g. an LV95 destination grid built with ``np.arange``), which gave wrong values or NaN
* fixed ``create_dst_grid`` with ``topright`` and ``resolution``: the resolution is now exact and ``topright``
  a closed upper bound, instead of the grid spacing being stretched to end exactly on ``topright``

1.0.0
-----

First open-source release of ``grid-process``, the continuation of the internal MeteoSwiss library
``gridefix-process`` (last internal version: 6.1.0).

**Breaking Change** (compared to ``gridefix-process`` 6.1.0)

* renamed the package from ``gridefix-process`` to ``grid-process`` (import name ``grid_process``)
* renamed the environment variable prefix from ``GRIDEFIX_PROCESS__`` to ``GRID_PROCESS__``
* versioning restarts at ``1.0.0``
