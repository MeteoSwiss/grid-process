# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the optimization module"""

import multiprocessing
import os
import threading
import warnings

import numpy as np
import pytest

from grid_process.optimization import memoize


class Counter:
    """Counts the calls of the decorated function and of the key function"""

    def __init__(self):
        self.func = 0
        self.key = 0

    def square(self, arr):
        self.func += 1
        return arr**2

    def key_of(self, arr):
        self.key += 1
        return arr.tobytes()


def test_memoize_key():
    """
    With a key function: results are cached by key and the decorated function is called with its usual arguments
    """

    counter = Counter()
    square = memoize(2, key=counter.key_of)(counter.square)

    np.testing.assert_array_equal(square(np.array([1, 2])), [1, 4])
    np.testing.assert_array_equal(square(np.array([1, 2])), [1, 4])
    assert counter.func == 1
    assert counter.key == 2
    np.testing.assert_array_equal(square(np.array([3])), [9])
    assert counter.func == 2


def test_memoize_lru():
    """
    The least recently used result is evicted, a cache hit refreshing its result
    """

    counter = Counter()
    square = memoize(2, key=counter.key_of)(counter.square)

    a, b, c = np.array([1]), np.array([2]), np.array([3])
    square(a)
    square(b)
    square(a)  # hit: b is now the least recently used
    square(c)  # evicts b
    assert counter.func == 3
    square(a)
    assert counter.func == 3
    square(b)
    assert counter.func == 4


def test_memoize_size_zero():
    """
    With a cache size of 0 and a key function, the function is returned undecorated: no key, no caching
    """

    counter = Counter()
    square = memoize(0, key=counter.key_of)(counter.square)
    assert square is counter.square or square.__func__ is counter.square.__func__

    square(np.array([1]))
    square(np.array([1]))
    assert counter.func == 2
    assert counter.key == 0


def test_memoize_hkey():
    """
    Without a key function, the key is the hkey keyword argument given by the caller, removed before the call
    """

    counter = Counter()
    square = memoize(2)(counter.square)

    np.testing.assert_array_equal(square(np.array([2]), hkey="a"), [4])
    square(np.array([2]), hkey="a")
    assert counter.func == 1
    # no hkey or hkey None: not cached
    square(np.array([2]))
    square(np.array([2]), hkey=None)
    assert counter.func == 3

    # size 0: hkey removed, not cached
    no_cache = memoize(0)(counter.square)
    no_cache(np.array([2]), hkey="a")
    no_cache(np.array([2]), hkey="a")
    assert counter.func == 5


def test_memoize_wraps():
    """
    The decorated function keeps the name and docstring of the original one
    """

    def func(x):
        """docstring"""
        return x

    for decorated in (memoize(2)(func), memoize(2, key=lambda x: x)(func), memoize(0)(func)):
        assert decorated.__name__ == "func"
        assert decorated.__doc__ == "docstring"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX only")
def test_memoize_fork_while_lock_held():
    """
    A process forked while another thread is inside the cache (holding its lock) can use the cache: the fork waits
    until the lock is released, instead of the child inheriting a held lock and blocking forever
    """

    holding, release = threading.Event(), threading.Event()

    class BlockingKey:
        """Key whose hash (computed while the cache lock is held) blocks until released"""

        def __hash__(self):
            holding.set()
            release.wait()
            return 0

    def identity(value, key):  # pylint: disable=unused-argument
        return value

    cached = memoize(2, key=lambda value, key: key)(identity)

    def child():
        assert cached(2, "other key") == 2

    thread = threading.Thread(target=cached, args=(1, BlockingKey()))
    thread.start()
    process = None
    try:
        assert holding.wait(10)
        threading.Timer(0.2, release.set).start()
        process = multiprocessing.get_context("fork").Process(target=child)
        with warnings.catch_warnings():  # Python >= 3.12 warns about forking a process with threads
            warnings.simplefilter("ignore", DeprecationWarning)
            process.start()
        process.join(30)
        assert process.exitcode == 0
    finally:
        release.set()
        if process is not None and process.is_alive():
            process.kill()
        thread.join()
