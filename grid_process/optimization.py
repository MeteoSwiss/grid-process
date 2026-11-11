# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module for optimization of computations
"""

import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from functools import wraps
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def memoize(maxsize: int = 10, key: Callable[..., Hashable] | None = None) -> Callable[[F], F]:
    """
    In-memory least recently used (LRU) cache decorator for functions with arbitrary (e.g. unhashable numpy array)
    arguments. The cache key is either computed by the ``key`` function or given by the caller.

    Parameters
    ----------
    maxsize : int, optional
        maximum number of cached results. When exceeded, the least recently used result is evicted. With
        ``maxsize <= 0``, nothing is cached and the decorator adds no overhead. Default is 10.
    key : callable, optional
        function called with the arguments of the decorated function, returning the (hashable) cache key, e.g.
        ``lambda arr, k: (arr.tobytes(), k)``. It is only called when caching is active (``maxsize > 0``), and
        the decorated function is then called with its usual arguments.
        If not given, the key is the optional keyword argument ``hkey`` passed at each call
        (e.g. ``func(arr, hkey=my_key)``), which is removed before calling the function. Calls without ``hkey``
        (or with ``hkey=None``) are not cached.

    Returns
    -------
    callable
        the decorator. Each decorated function has its own cache, kept in memory until the end of the process.
        The cache can be used from several threads. Each process has its own cache: a process created by ``fork``
        starts with a copy of the parent's cache (the fork waits until no thread is reading or writing it), one
        created by ``spawn`` or ``forkserver`` with an empty cache.

    Note
    ----
    The cache size of the functions cached by grid-process is read when the module is imported, from
    ``grid_process/config/grid_process.yaml`` or the environment variables ``GRID_PROCESS__cache_size__<name>``
    (e.g. ``GRID_PROCESS__cache_size__nn_lookup=0`` to deactivate the cache of nearest neighbor lookups).
    """

    def real_memoize(func: F) -> F:
        if maxsize <= 0:
            if key is not None:
                return func

            @wraps(func)
            def _no_cache(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop("hkey", None)
                return func(*args, **kwargs)

            return cast(F, _no_cache)

        cache: OrderedDict[Hashable, Any] = OrderedDict()
        # held only for the cache reads and writes: nothing slow (e.g. func) or forking may run while it is held
        lock = threading.Lock()
        if hasattr(os, "register_at_fork"):  # POSIX only
            # a fork waits until no other thread holds the lock: the child gets a consistent cache and a free lock
            os.register_at_fork(before=lock.acquire, after_in_parent=lock.release, after_in_child=lock.release)

        def _cached_call(hkey: Hashable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if hkey is not None:
                with lock:
                    hit = hkey in cache
                    if hit:
                        cache.move_to_end(hkey)
                        result = cache[hkey]
                if hit:
                    logger.debug("@memoize: %s cached!", func.__name__)
                    return result
            logger.debug("@memoize: %s computing ...", func.__name__)
            result = func(*args, **kwargs)
            if hkey is not None:
                with lock:
                    cache[hkey] = result
                    cache.move_to_end(hkey)
                    while len(cache) > maxsize:
                        cache.popitem(last=False)
                    size = len(cache)
                logger.debug("@memoize: %s cache size is %s", func.__name__, size)
            return result

        if key is not None:
            get_key = key

            @wraps(func)
            def _memoize_key(*args: Any, **kwargs: Any) -> Any:
                return _cached_call(get_key(*args, **kwargs), args, kwargs)

            return cast(F, _memoize_key)

        @wraps(func)
        def _memoize_hkey(*args: Any, **kwargs: Any) -> Any:
            hkey = kwargs.pop("hkey", None)
            return _cached_call(hkey, args, kwargs)

        return cast(F, _memoize_hkey)

    return real_memoize
