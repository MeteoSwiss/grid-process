.. image:: https://img.shields.io/pypi/v/grid-process.svg
    :target: https://pypi.org/project/grid-process/

.. image:: https://img.shields.io/pypi/pyversions/grid-process.svg
    :target: https://pypi.org/project/grid-process/

.. image:: https://img.shields.io/pypi/l/grid-process.svg
    :target: https://pypi.org/project/grid-process/

.. image:: https://github.com/MeteoSwiss/grid-process/actions/workflows/CI_test.yaml/badge.svg
    :target: https://github.com/MeteoSwiss/grid-process/actions/workflows/CI_test.yaml

.. image:: https://github.com/MeteoSwiss/grid-process/actions/workflows/CI_publish_dev_documentation.yaml/badge.svg
    :target: https://github.com/MeteoSwiss/grid-process/actions/workflows/CI_publish_dev_documentation.yaml

===============
Getting Started
===============

Weather forecast oriented Python library to process xarray Datasets.

.. warning::

    **Project status and deprecation outlook.** ``grid-process`` mainly serves the needs of MeteoSwiss operational
    processing. As equivalent functionality becomes available in the
    `ECMWF earthkit ecosystem <https://earthkit.ecmwf.int/>`_, parts of this library, potentially all of it, will be
    gradually deprecated in favour of earthkit. Any deprecation will be announced in the changelog. If you start a new
    project, please consider earthkit first.

Features
--------

* Grid to points interpolations routines
* Temporal interpolation routines
* Spatial and temporal aggregation routines
* Ensemble and spatial statistics
* Configurable data model (dimension, coordinate and variable names) to support many data conventions
* And much more

.. note::

    ``grid-process`` was originally developed internally at MeteoSwiss as ``gridefix-process`` (versions 1 to 6),
    to process grids from an internal grid database. The first open-source release is ``1.0.0``.
    See `Migrating from gridefix-process`_ if you are upgrading from the internal package.

Development Setup with Poetry
-----------------------------

Building the Project
''''''''''''''''''''
.. code-block:: console

    $ cd grid-process
    $ poetry install

Run Tests
'''''''''

.. code-block:: console

    $ poetry run pytest

Run Quality Tools
'''''''''''''''''

.. code-block:: console

    $ poetry run pylint grid_process
    $ poetry run mypy grid_process
    $ poetry run ruff format

Generate Documentation
''''''''''''''''''''''

.. code-block:: console

    $ poetry run sphinx-build doc doc/_build

Then open the index.html file generated in *grid-process/doc/_build/*.

Build wheels
''''''''''''

.. code-block:: console

    $ poetry build

Using the Library
-----------------

To install grid-process in your project, run this command in your terminal:

.. code-block:: console

    $ poetry add grid-process

You can then use the library in your project through

.. code-block:: python

    import grid_process

Migrating from gridefix-process
-------------------------------

* Package and import name: ``gridefix-process`` / ``gridefix_process`` became ``grid-process`` / ``grid_process``.
* Environment variable prefix: ``GRIDEFIX_PROCESS__`` became ``GRID_PROCESS__``
  (e.g. ``GRID_PROCESS__data_model__validtime``).
* Versioning restarts at ``1.0.0``; the last internal release was ``gridefix-process`` 6.1.0.

Release the Project
-------------------

The project follows the **GitOps concept**: releases are triggered whenever a Git TAG is created.

The TAG must follow the `semantic version <https://semver.org/>`__ format and `PEP 440 <https://peps.python.org/pep-0440/>`__ , otherwise the release task will fail.

Follow these steps to create a new release:

* Adapt CHANGELOG.rst with release information
* Adapt ``doc/_static/switcher_config.json`` adding the new documentation URL for the release
* Create a new Release in the Github project
