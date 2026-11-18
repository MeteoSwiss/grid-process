# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sparse matrix helpers shared by the grid_interp and ens_stats modules (package internal)
"""

import numpy as np
from scipy import sparse


def sparse_matmul(weights: list[sparse.csr_array], values: np.ndarray) -> np.ndarray:
    """
    Matrix products of each matrix of ``weights`` along the corresponding first axis of ``values``, copied into
    C-contiguous rows of points if needed (e.g. data stored with the spatial dimensions last, or sliced): a product
    per field would avoid the copy but is slower (it reads the matrix once per field) and keeps the copy of a sliced
    source.
    """

    for axis, weight in enumerate(weights):
        moved = np.moveaxis(values, axis, 0)
        out = weight.astype(values.dtype, copy=False) @ moved.reshape(moved.shape[0], -1)
        values = np.moveaxis(out.reshape(weight.shape[:1] + moved.shape[1:]), 0, axis)
    return values


def rows_matrix(weights: np.ndarray, index: np.ndarray, n_columns: int) -> sparse.csr_array:
    """
    CSR matrix of shape (n_rows, ``n_columns``) with the ``weights`` (n_rows, k) at the columns ``index`` (n_rows, k)
    of each row, with int32 indices when possible: int32 ``index`` (e.g. from
    :func:`~grid_process.grid_interp.nn_lookup_cached`) not copied.
    """

    dtype = np.int32 if max(index.size, n_columns) < 2**31 else np.int64
    return sparse.csr_array(
        (
            weights.ravel(),
            index.ravel().astype(dtype, copy=False),
            np.arange(0, index.size + 1, index.shape[1], dtype=dtype),
        ),
        shape=(len(index), n_columns),
    )
