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


def test_ds_to_da_no_copy(model_data):
    """
    With copy=False, a single variable gives the same dataarray as to_array, as a view of its data. Several
    variables, or a variable not spanning all dimensions, are stacked (copied)
    """

    single = model_data[[next(iter(model_data.data_vars))]].assign_attrs(source="test")
    name = next(iter(single.data_vars))
    da_view, vars_attrs = dsu.ds_to_da(single, copy=False)

    assert da_view.identical(single.to_array())
    assert vars_attrs == {name: single[name].attrs}
    assert np.shares_memory(da_view.values, single[name].values)
    assert not np.shares_memory(dsu.ds_to_da(single)[0].values, single[name].values)  # default: copy

    if len(model_data.data_vars) > 1:
        da_stack, _ = dsu.ds_to_da(model_data, copy=False)
        assert da_stack.identical(model_data.to_array())
        assert not np.shares_memory(da_stack.values, model_data[name].values)

    other_dim = single.assign_coords(extra=("extra", [1, 2]))  # a dimension of a coordinate only
    da_other, _ = dsu.ds_to_da(other_dim, copy=False)
    assert da_other.identical(other_dim.to_array())


def test_cast_dtype(model_data):
    """Simple testcase for the cast_dtype function from ds_utils module"""

    ds_out = model_data.astype(np.float64)
    dsu.cast_dtype(ds_out, model_data)
    dtypes_in = {name: da.dtype for name, da in model_data.items()}
    dtypes_out = {name: da.dtype for name, da in ds_out.items()}

    assert dtypes_in == dtypes_out
