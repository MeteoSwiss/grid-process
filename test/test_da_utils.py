# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the da_utils module"""

import numpy as np
import grid_process.da_utils as dau
import grid_process.ds_utils as dsu


def test_da_to_ds(model_data):
    """Simple testcase for the da_to_ds function from da_utils module"""

    da_out, vars_attrs = dsu.ds_to_da(model_data)
    ds_out = dau.da_to_ds(da_out, vars_attrs)

    assert list(ds_out.data_vars) == list(model_data.data_vars)
    assert all([np.allclose(ds_out[var], model_data[var]) for var in vars_attrs])
    assert all([ds_out[var].attrs == model_data[var].attrs for var in vars_attrs])


def test_concat_str_arrays():
    """Simple testcase for the concat_str_arrays function from da_utils module"""
    pass


def test_bin_to_class():
    """Simple testcase for the bin_to_class function from da_utils module"""
    pass
