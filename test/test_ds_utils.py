# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"Tests for the ds_utils module"

import numpy as np

import grid_process.ds_utils as dsu


def test_ds_to_da(model_data):
    """Simple testcase for the ds_to_da function from ds_utils module"""

    expected_attrs = {var: da.attrs for var, da in model_data.items()}
    expected_dims = set(tuple(model_data.dims) + ("variable",))
    da_out, vars_attrs = dsu.ds_to_da(model_data)

    assert expected_attrs == vars_attrs
    assert expected_dims == set(da_out.dims)


def test_cast_dtype(model_data):
    """Simple testcase for the cast_dtype function from ds_utils module"""

    ds_out = model_data.astype(np.float64)
    dsu.cast_dtype(ds_out, model_data)
    dtypes_in = {name: da.dtype for name, da in model_data.items()}
    dtypes_out = {name: da.dtype for name, da in ds_out.items()}

    assert dtypes_in == dtypes_out
